#pragma once

#include "point_cloud.hpp"
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>

namespace pc_eval {

class PLYLoader {
public:
    static bool load(const std::string& filename, PointCloud& pc) {
        
        std::ifstream file(filename, std::ios::binary);
        std::string line;
        std::getline(file, line); // ply
        
        std::string format;
        size_t vertex_count = 0;
        std::vector<Property> properties;
        bool is_binary = false;

        while (std::getline(file, line)) {
            // Trim whitespace
            line.erase(0, line.find_first_not_of(" \t\r\n"));
            line.erase(line.find_last_not_of(" \t\r\n") + 1);

            if (line.empty()) continue;

            std::stringstream ss(line);
            std::string token;
            ss >> token;

            if (token == "format") {
                ss >> format;
                is_binary = (format == "binary_little_endian");
            } else if (token == "element") {
                std::string type;
                ss >> type;
                if (type == "vertex") {
                    ss >> vertex_count;
                }
            } else if (token == "property") {
                std::string type, name;
                ss >> type >> name;
                properties.push_back({name, type});
            } else if (token == "end_header") {
                break;
            }
        }

        pc.points.clear();
        pc.points.reserve(vertex_count);

        if (is_binary) {
            size_t vertex_size = 0;
            int x_idx = -1, y_idx = -1, z_idx = -1;
            std::vector<size_t> prop_offsets;
            std::vector<size_t> prop_sizes;

            for (size_t i = 0; i < properties.size(); ++i) {
                prop_offsets.push_back(vertex_size);
                size_t size = 0;
                if (properties[i].type == "float") size = 4;
                else if (properties[i].type == "double") size = 8;
                else if (properties[i].type == "int" || properties[i].type == "uint") size = 4;
                else if (properties[i].type == "char" || properties[i].type == "uchar") size = 1;
                else if (properties[i].type == "short" || properties[i].type == "ushort") size = 2;
                
                prop_sizes.push_back(size);
                vertex_size += size;

                if (properties[i].name == "x") x_idx = (int)i;
                if (properties[i].name == "y") y_idx = (int)i;
                if (properties[i].name == "z") z_idx = (int)i;
            }

            if (x_idx == -1 || y_idx == -1 || z_idx == -1) return false;

            std::vector<char> buffer(vertex_size);
            for (size_t i = 0; i < vertex_count; ++i) {
                file.read(buffer.data(), vertex_size);
                Point3 p;
                
                auto read_float = [&](int idx) {
                    float val = 0;
                    if (properties[idx].type == "float") {
                        std::copy(buffer.data() + prop_offsets[idx], buffer.data() + prop_offsets[idx] + 4, reinterpret_cast<char*>(&val));
                    } else if (properties[idx].type == "double") {
                        double dval;
                        std::copy(buffer.data() + prop_offsets[idx], buffer.data() + prop_offsets[idx] + 8, reinterpret_cast<char*>(&dval));
                        val = (float)dval;
                    }
                    return val;
                };

                p.x = read_float(x_idx);
                p.y = read_float(y_idx);
                p.z = read_float(z_idx);
                pc.points.push_back(p);
            }
        } else {
            // ASCII
            for (size_t i = 0; i < vertex_count; ++i) {
                std::getline(file, line);
                std::stringstream ss(line);
                Point3 p;
                // This assumes x, y, z are the first 3 properties. 
                // To be robust, we should read according to properties list.
                std::vector<float> values;
                float v;
                while (ss >> v) values.push_back(v);
                
                int x_idx = -1, y_idx = -1, z_idx = -1;
                for(int j=0; j<properties.size(); ++j) {
                    if(properties[j].name == "x") x_idx = j;
                    if(properties[j].name == "y") y_idx = j;
                    if(properties[j].name == "z") z_idx = j;
                }
                
                if (x_idx < values.size() && y_idx < values.size() && z_idx < values.size()) {
                    p.x = values[x_idx];
                    p.y = values[y_idx];
                    p.z = values[z_idx];
                    pc.points.push_back(p);
                }
            }
        }

        return true;
    }

private:
    struct Property {
        std::string name;
        std::string type; // float, double, int, etc.
    };


};

} // namespace pc_eval
