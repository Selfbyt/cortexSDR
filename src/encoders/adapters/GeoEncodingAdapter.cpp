#include "GeoEncodingAdapter.hpp"

GeoEncodingAdapter::GeoEncodingAdapter() : encoder_() {}

std::vector<size_t> GeoEncodingAdapter::encode(double latitude, double longitude) const {
    return encoder_.encode(latitude, longitude);
}

std::pair<double, double> GeoEncodingAdapter::decode(const std::vector<size_t>& indices) const {
    return encoder_.decode(indices);
}
