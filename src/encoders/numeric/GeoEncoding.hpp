#pragma once
#include <vector>
#include <utility>

class GeoEncoding {
public:
    GeoEncoding(double latMin = -90.0, double latMax = 90.0, double lonMin = -180.0, double lonMax = 180.0, size_t buckets = 1000);
    std::vector<size_t> encode(double latitude, double longitude) const;
    std::pair<double, double> decode(const std::vector<size_t>& indices) const;
private:
    double latMin_, latMax_, lonMin_, lonMax_;
    size_t buckets_;
};
