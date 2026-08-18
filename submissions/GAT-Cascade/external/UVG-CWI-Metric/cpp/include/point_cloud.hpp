#pragma once

#include <vector>
#include <array>
#include <cmath>

namespace pc_eval {

struct Point3 {
    float x, y, z;

    float distance_sq(const Point3& other) const {
        float dx = x - other.x;
        float dy = y - other.y;
        float dz = z - other.z;
        return dx * dx + dy * dy + dz * dz;
    }

    float distance(const Point3& other) const {
        return std::sqrt(distance_sq(other));
    }
};

class PointCloud {
public:
    std::vector<Point3> points;

    void clear() { points.clear(); }
    size_t size() const { return points.size(); }
    bool empty() const { return points.empty(); }
    void add(const Point3& p) { points.push_back(p); }

    const Point3& operator[](size_t i) const { return points[i]; }
};

} // namespace pc_eval
