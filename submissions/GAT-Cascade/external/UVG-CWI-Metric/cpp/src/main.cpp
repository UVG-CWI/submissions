#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <random>
#include <fstream>
#include <iomanip>
#include <chrono>
#include "point_cloud.hpp"
#include "ply_loader.hpp"
#include "metrics.hpp"

namespace fs = std::filesystem;
using namespace pc_eval;

struct Config {
    std::string mode;
    std::string sequence_name = "output";
    std::string cg_dir = "dataset/UVG-CWI-DQPC/OrangeKettlebell/CG/15fps";
    std::string he_dir = "dataset/UVG-CWI-DQPC/OrangeKettlebell/HE/15fps";
    std::string reconstructed_dir = "dataset/UVG-CWI-DQPC/OrangeKettlebell/Reconstructed";
    int max_frames = -1;
    int sample_points = 200000;
};

void print_usage() {
    std::cout << "Usage: evaluate_points --mode <baseline|reconstructed> --sequence_name <name> [options]\n"
              << "Options:\n"
              << "  --sequence_name <name>   Name of the sequence (used for output filename)\n"
              << "  --cg_dir <path>          Low-quality CG directory\n"
              << "  --he_dir <path>          Ground truth HE directory\n"
              << "  --reconstructed_dir <path>  Reconstructed directory\n"
              << "  --max_frames <n>         Maximum number of frames to evaluate\n"
              << "  --sample_points <n>      Number of points to sample (default: 200000)\n";
}

int main(int argc, char* argv[]) {
    Config config;
    if (argc < 2) {
        print_usage();
        return 1;
    }

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--mode" && i + 1 < argc) config.mode = argv[++i];
        else if (arg == "--sequence_name" && i + 1 < argc) config.sequence_name = argv[++i];
        else if (arg == "--cg_dir" && i + 1 < argc) config.cg_dir = argv[++i];
        else if (arg == "--he_dir" && i + 1 < argc) config.he_dir = argv[++i];
        else if (arg == "--reconstructed_dir" && i + 1 < argc) config.reconstructed_dir = argv[++i];
        else if (arg == "--max_frames" && i + 1 < argc) config.max_frames = std::stoi(argv[++i]);
        else if (arg == "--sample_points" && i + 1 < argc) config.sample_points = std::stoi(argv[++i]);
    }

    if (config.mode != "baseline" && config.mode != "reconstructed") {
        std::cerr << "Error: --mode must be 'baseline' or 'reconstructed'\n";
        return 1;
    }

    fs::path test_path = (config.mode == "baseline") ? config.cg_dir : config.reconstructed_dir;
    fs::path gt_path = config.he_dir;
    std::string suffix = (config.mode == "reconstructed") ? "_pointcloud.ply" : ".ply";

    std::vector<fs::path> test_files;
    if (fs::exists(test_path)) {
        for (const auto& entry : fs::directory_iterator(test_path)) {
            if (entry.path().extension() == ".ply" && 
                (config.mode == "baseline" || entry.path().filename().string().find(suffix) != std::string::npos)) {
                test_files.push_back(entry.path());
            }
        }
    }
    std::sort(test_files.begin(), test_files.end());

    std::vector<fs::path> gt_files;
    if (fs::exists(gt_path)) {
        for (const auto& entry : fs::directory_iterator(gt_path)) {
            if (entry.path().extension() == ".ply") {
                gt_files.push_back(entry.path());
            }
        }
    }
    std::sort(gt_files.begin(), gt_files.end());

    size_t num_pairs = std::min(test_files.size(), gt_files.size());
    if (config.max_frames > 0) num_pairs = std::min(num_pairs, (size_t)config.max_frames);

    if (num_pairs == 0) {
        std::cerr << "Error: No matching pairs found in paths:\n"
                  << "  Test: " << test_path << "\n"
                  << "  GT:   " << gt_path << "\n";
        return 1;
    }

    std::cout << "\nEvaluating " << num_pairs << " pairs in " << config.mode << " mode\n";
    
    std::vector<float> thresholds = {10.0f, 20.0f, 30.0f, 50.0f};
    std::vector<Metrics> all_results;
    
    fs::create_directories("results");
    std::string output_csv = "results/" + config.sequence_name + "_" + config.mode + "_metrics.csv";
    
    std::ofstream csv(output_csv);
    csv << "frame,test_file,gt_file,chamfer_distance,accuracy,completeness,hausdorff_distance,"
        << "precision_10.0,recall_10.0,fscore_10.0,"
        << "precision_20.0,recall_20.0,fscore_20.0,"
        << "precision_30.0,recall_30.0,fscore_30.0,"
        << "precision_50.0,recall_50.0,fscore_50.0\n";

    for (size_t i = 0; i < num_pairs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        std::cout << "[" << i + 1 << "/" << num_pairs << "] Processing frame " << i << "...\n";
        
        // Load point clouds
        auto start_load = std::chrono::high_resolution_clock::now();

        PointCloud pc_test_full, pc_gt_full;
        if (!PLYLoader::load(test_files[i].string(), pc_test_full)) {
            std::cerr << "Failed to load test file: " << test_files[i] << "\n";
            continue;
        }
        if (!PLYLoader::load(gt_files[i].string(), pc_gt_full)) {
             std::cerr << "Failed to load GT file: " << gt_files[i] << "\n";
             continue;
        }

        auto end_load = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_load = end_load - start_load;
        std::cout << "    Load Time: " << elapsed_load.count() << "s" << std::endl;

        Metrics m = MetricsCalculator::compute(pc_test_full, pc_gt_full, thresholds);
        all_results.push_back(m);

        csv << i << "," << test_files[i].filename().string() << "," << gt_files[i].filename().string() << ","
            << std::fixed << std::setprecision(6) << m.chamfer_distance << "," << m.accuracy << "," << m.completeness << "," << m.hausdorff_distance;
        
        for (float t : thresholds) {
            std::string ts = std::to_string((int)t);
            csv << "," << m.precision_recall_fscore["precision_" + ts]
                << "," << m.precision_recall_fscore["recall_" + ts]
                << "," << m.precision_recall_fscore["fscore_" + ts];
        }
        csv << "\n";

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;

        std::cout << "    Chamfer: " << m.chamfer_distance << "\n"
                  << "    F-score@10mm: " << m.precision_recall_fscore["fscore_10"] << "\n"
                  << "    Time: " << diff.count() << "s\n\n";
    }

    if (!all_results.empty()) {
        double avg_chamfer = 0, avg_f10 = 0, avg_f20 = 0;
        for (const auto& r : all_results) {
            avg_chamfer += r.chamfer_distance;
            avg_f10 += r.precision_recall_fscore.at("fscore_10");
            avg_f20 += r.precision_recall_fscore.at("fscore_20");
        }
        avg_chamfer /= all_results.size();
        avg_f10 /= all_results.size();
        avg_f20 /= all_results.size();

        std::cout << "\nSummary (" << all_results.size() << " frames):\n"
                  << "  Average Chamfer Distance: " << avg_chamfer << "\n"
                  << "  Average F-score@10mm:     " << avg_f10 << "\n"
                  << "  Average F-score@20mm:     " << avg_f20 << "\n";
    }

    std::cout << "\nResults saved to " << output_csv << "\n";

    return 0;
}
