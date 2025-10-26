#pragma once
#include "../numeric/GeoEncoding.hpp"
#include <vector>
#include <utility>

class GeoEncodingAdapter {
public:
    GeoEncodingAdapter();
    std::vector<size_t> encode(double latitude, double longitude) const;
    std::pair<double, double> decode(const std::vector<size_t>& indices) const;
private:
    GeoEncoding encoder_;
};
