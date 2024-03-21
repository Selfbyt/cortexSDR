#ifndef DATE_TIME_ENCODING_HPP
#define DATE_TIME_ENCODING_HPP

#include <string>
#include <vector>

class DateTimeEncoding {
public:
    std::vector<size_t> encodeDateTime(const std::string& dateTime) const;
};

#endif // DATE_TIME_ENCODING_HPP
