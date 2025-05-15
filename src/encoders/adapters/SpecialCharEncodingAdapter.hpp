#pragma once
#include "../text/SpecialCharEncoding.hpp"
#include <vector>
#include <string>

class SpecialCharEncodingAdapter {
public:
    SpecialCharEncodingAdapter();
    std::vector<size_t> encode(const std::string& text) const;
    std::string decode(const std::vector<size_t>& indices) const;
private:
    SpecialCharEncoding encoder_;
};
