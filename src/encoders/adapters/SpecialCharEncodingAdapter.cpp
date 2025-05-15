#include "SpecialCharEncodingAdapter.hpp"

SpecialCharEncodingAdapter::SpecialCharEncodingAdapter() {}

std::vector<size_t> SpecialCharEncodingAdapter::encode(const std::string& text) const {
    return encoder_.encode(text);
}

std::string SpecialCharEncodingAdapter::decode(const std::vector<size_t>& indices) const {
    return encoder_.decode(indices);
}
