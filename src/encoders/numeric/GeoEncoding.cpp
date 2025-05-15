#include "GeoEncoding.hpp"
#include <cmath>
#include <vector>
#include <algorithm>
#include <string>
#include <sstream>
#include <stdexcept>

GeoEncoding::GeoEncoding(double latMin, double latMax, double lonMin, double lonMax, size_t buckets)
    : latMin_(latMin), latMax_(latMax), lonMin_(lonMin), lonMax_(lonMax), buckets_(buckets) {}

std::vector<size_t> GeoEncoding::encode(double latitude, double longitude) const {
    // Clamp and quantize
    latitude = std::clamp(latitude, latMin_, latMax_);
    longitude = std::clamp(longitude, lonMin_, lonMax_);
    size_t latIdx = static_cast<size_t>(((latitude - latMin_) / (latMax_ - latMin_)) * (buckets_ - 1));
    size_t lonIdx = static_cast<size_t>(((longitude - lonMin_) / (lonMax_ - lonMin_)) * (buckets_ - 1));
    return {latIdx, lonIdx};
}

std::pair<double, double> GeoEncoding::decode(const std::vector<size_t>& indices) const {
    if (indices.size() < 2) throw std::invalid_argument("Not enough indices for geo decode");
    double lat = latMin_ + (static_cast<double>(indices[0]) / (buckets_ - 1)) * (latMax_ - latMin_);
    double lon = lonMin_ + (static_cast<double>(indices[1]) / (buckets_ - 1)) * (lonMax_ - lonMin_);
    return {lat, lon};
}
