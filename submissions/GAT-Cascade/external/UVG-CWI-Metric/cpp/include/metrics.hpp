#pragma once

#include "point_cloud.hpp"
#include "kdtree.hpp"
#include <map>
#include <string>
#include <vector>
#include <iostream>
#include <numeric>
#include <tuple>
#include <cmath>
#include <omp.h>
#include <execution>
#include <algorithm>

namespace pc_eval {

struct Metrics {
    double chamfer_distance;
    double accuracy;
    double completeness;
    double hausdorff_distance;
    std::map<std::string, double> precision_recall_fscore;
};

class MetricsCalculator {
public:
    static Metrics compute(const PointCloud& pc1_full, const PointCloud& pc2_full, 
                           const std::vector<float>& thresholds) {
        Metrics metrics;

        if (pc1_full.empty() || pc2_full.empty()) {
            return metrics;
        }

        // 1. Build trees from FULL clouds
        KDTree tree1_full, tree2_full;
        tree1_full.build(pc1_full);
        tree2_full.build(pc2_full);

        // 2. Compute Precision/Recall/Chamfer from FULL clouds with OpenMP
        std::vector<float> full_dists_1_to_2(pc2_full.size());
        #pragma omp parallel for
        for (long long i = 0; i < (long long)pc2_full.size(); ++i) {
            auto res = tree1_full.query(pc2_full[i]);
            full_dists_1_to_2[i] = std::sqrt(res.dist_sq);
        }

        std::vector<float> full_dists_2_to_1(pc1_full.size());
        #pragma omp parallel for
        for (long long i = 0; i < (long long)pc1_full.size(); ++i) {
            auto res = tree2_full.query(pc1_full[i]);
            full_dists_2_to_1[i] = std::sqrt(res.dist_sq);
        }

        // 3. Compute Aggregated Metrics (Chamfer, Hausdorff) from full distances
        // Upgraded to par_unseq to explicitly allow both threading and vectorization
        double sum_d_1_to_2 = std::reduce(std::execution::par_unseq, full_dists_1_to_2.begin(), full_dists_1_to_2.end());
        double sum_d_2_to_1 = std::reduce(std::execution::par_unseq, full_dists_2_to_1.begin(), full_dists_2_to_1.end());
        
        float max_d_1_to_2 = *std::max_element(std::execution::par_unseq, full_dists_1_to_2.begin(), full_dists_1_to_2.end());
        float max_d_2_to_1 = *std::max_element(std::execution::par_unseq, full_dists_2_to_1.begin(), full_dists_2_to_1.end());

        metrics.accuracy = sum_d_2_to_1 / pc1_full.size();
        metrics.completeness = sum_d_1_to_2 / pc2_full.size();
        metrics.chamfer_distance = (metrics.accuracy + metrics.completeness) / 2.0;
        metrics.hausdorff_distance = std::max(max_d_1_to_2, max_d_2_to_1);

        for (float t : thresholds) {
            auto count_below = [](const std::vector<float>& dists, float threshold) -> double {
                // unseq handles the explicit SIMD vectorization that AVX was previously providing
                return static_cast<double>(std::count_if(
                    std::execution::unseq, 
                    dists.begin(), 
                    dists.end(), 
                    [threshold](float d) { return d < threshold; }
                ));
            };

            double p = count_below(full_dists_2_to_1, t);
            p /= pc1_full.size();

            double r = count_below(full_dists_1_to_2, t);
            r /= pc2_full.size();

            double f = (p + r > 0) ? (2 * p * r / (p + r)) : 0;

            std::string ts = std::to_string(static_cast<int>(t));
            metrics.precision_recall_fscore["precision_" + ts] = p;
            metrics.precision_recall_fscore["recall_" + ts] = r;
            metrics.precision_recall_fscore["fscore_" + ts] = f;
        }

        return metrics;
    }
};

} // namespace pc_eval