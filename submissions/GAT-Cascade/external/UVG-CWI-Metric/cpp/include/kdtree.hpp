#pragma once

#include "point_cloud.hpp"
#include <vector>
#include <algorithm>
#include <limits>
#include <memory>
#include <immintrin.h>
#include <execution>
#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace pc_eval {

class KDTree {
public:
    struct Node {
        Point3 point;
        int left = -1;
        int right = -1;
    };

    void build(const PointCloud& pc) {
        nodes.clear();
        if (pc.empty()) return;

        nodes.reserve(pc.size());
        std::vector<const Point3*> ptrs;
        ptrs.reserve(pc.size());
        for (const auto& p : pc.points) {
            ptrs.push_back(&p);
        }

        root = build_recursive(ptrs.data(), pc.size(), 0);
    }

    struct QueryResult {
        float dist_sq = std::numeric_limits<float>::max();
        const Point3* point = nullptr;
    };

    QueryResult query(const Point3& p) const {
        QueryResult res;
        if (root != -1) {
            query_recursive(root, p, 0, res);
        }
        return res;
    }


private:
    std::vector<Node> nodes;
    int root = -1;

    int build_recursive(const Point3** ptrs, size_t count, int depth) {
        if (count == 0) return -1;

        int axis = depth % 3;
        auto cmp = [axis](const Point3* a, const Point3* b) {
            if (axis == 0) return a->x < b->x;
            if (axis == 1) return a->y < b->y;
            return a->z < b->z;
        };

        size_t mid = count / 2;
        
        // Use parallel nth_element for large chunks
        if (count > 8192) {
            std::nth_element(std::execution::par, ptrs, ptrs + mid, ptrs + count, cmp);
        } else {
            std::nth_element(ptrs, ptrs + mid, ptrs + count, cmp);
        }

        int node_idx = (int)nodes.size();
        nodes.push_back({*ptrs[mid], -1, -1});

        int left = build_recursive(ptrs, mid, depth + 1);
        int right = build_recursive(ptrs + mid + 1, count - mid - 1, depth + 1);
        
        nodes[node_idx].left = left;
        nodes[node_idx].right = right;

        return node_idx;
    }

    void query_recursive(int idx, const Point3& p, int depth, QueryResult& res) const {
        if (idx == -1) return;
        const auto& node = nodes[idx];

        float d_sq = node.point.distance_sq(p);
        if (d_sq < res.dist_sq) {
            res.dist_sq = d_sq;
            res.point = &node.point;
        }

        int axis = depth % 3;
        float diff = 0;
        if (axis == 0) diff = p.x - node.point.x;
        else if (axis == 1) diff = p.y - node.point.y;
        else diff = p.z - node.point.z;

        int near = diff < 0 ? node.left : node.right;
        int far = diff < 0 ? node.right : node.left;

        query_recursive(near, p, depth + 1, res);

        if (diff * diff < res.dist_sq) {
            query_recursive(far, p, depth + 1, res);
        }
    }
};

} // namespace pc_eval
