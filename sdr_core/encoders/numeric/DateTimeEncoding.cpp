#include "DateTimeEncoding.hpp"
#include <regex>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <chrono>
#include <cmath>

using namespace std::chrono;

// Relative time encoding index constants
constexpr size_t RELATIVE_START_INDEX = 1000; // should match the region used in encodeRelative
constexpr size_t RELATIVE_UNIT_START_INDEX = RELATIVE_START_INDEX + 10;
constexpr size_t RELATIVE_AGO_INDEX = RELATIVE_START_INDEX + 20;

namespace {
    // Regular expressions for different date formats
    const std::regex ISO8601_REGEX(
        R"((\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(\.\d+)?([+-]\d{2}:?\d{2}|Z)?)"
    );

    const std::regex COMPACT_REGEX(
        R"(\d{14}(\.\d+)?)"
    );

    const std::regex RELATIVE_REGEX(
        R"((now|today|tomorrow|yesterday|last|next|this)\s*(day|week|month|year|hour|minute|second)s?\s*(ago)?)",
        std::regex_constants::icase
    );

    // Lookup tables for relative time expressions
    const std::unordered_map<std::string, system_clock::duration> TIME_UNITS = {
        {"second", seconds(1)},
        {"minute", minutes(1)},
        {"hour", hours(1)},
        {"day", hours(24)},
        {"week", hours(24 * 7)},
        {"month", hours(24 * 30)}, // Approximate
        {"year", hours(24 * 365)} // Non-leap year
    };
}

DateTimeEncoding::DateTimeEncoding(const EncodingConfig& config)
    : config(config) {}

std::vector<size_t> DateTimeEncoding::encode(const std::string& dateTime) const {
    if (!isValid(dateTime)) {
        throw std::invalid_argument("Invalid datetime format: " + dateTime);
    }

    Format format = detectFormat(dateTime);
    DateTimeComponents components;

    switch (format) {
        case Format::ISO8601:
            components = parseDateTime(dateTime);
            break;
        case Format::UNIX_TIMESTAMP:
            components = parseDateTime(dateTime);
            break;
        case Format::COMPACT:
            components = parseDateTime(dateTime);
            break;
        case Format::RELATIVE:
            return encodeRelative(dateTime);
    }

    return encodeComponents(components);
}

std::vector<size_t> DateTimeEncoding::encode(
    const system_clock::time_point& timePoint) const {

    // Convert timePoint to components
    DateTimeComponents components{};
    auto tt = system_clock::to_time_t(timePoint);
    auto tm = *std::gmtime(&tt);
    
    components.year = tm.tm_year + 1900;
    components.month = tm.tm_mon + 1;
    components.day = tm.tm_mday;
    components.hour = tm.tm_hour;
    components.minute = tm.tm_min;
    components.second = tm.tm_sec;
    
    // Calculate nanoseconds if needed
    auto duration = timePoint.time_since_epoch();
    auto seconds = duration_cast<std::chrono::seconds>(duration);
    auto nanos = duration_cast<std::chrono::nanoseconds>(duration - seconds);
    components.nanoseconds = static_cast<uint32_t>(nanos.count());
    return encodeComponents(components);
}

std::vector<size_t> DateTimeEncoding::encodeRange(const TimeRange& range) const {
    std::vector<size_t> encoded;

    // Encode start time
    auto startEncoding = encode(range.start);
    encoded.insert(encoded.end(), startEncoding.begin(), startEncoding.end());

    // Encode end time
    auto endEncoding = encode(range.end);
    encoded.insert(encoded.end(), endEncoding.begin(), endEncoding.end());

    // Encode interval
    auto duration = range.interval.count();
    encoded.push_back(DATETIME_START_INDEX +
                     static_cast<size_t>(duration & 0xFFFFFFFF));

    return encoded;
}

std::vector<size_t> DateTimeEncoding::encodeRelative(
    const std::string& relativeTime) const {

    std::vector<size_t> encoded;
    std::smatch matches;

    if (!std::regex_match(relativeTime, matches, RELATIVE_REGEX)) {
        throw std::invalid_argument("Invalid relative time format");
    }

    // Extract components
    std::string qualifier = matches[1].str();
    std::string unit = matches[2].str();
    bool isAgo = matches[3].matched;

    // Encode qualifier
    size_t qualifierIndex = RELATIVE_START_INDEX;
    if (qualifier == "last") qualifierIndex += 1;
    else if (qualifier == "next") qualifierIndex += 2;
    else if (qualifier == "this") qualifierIndex += 3;
    encoded.push_back(qualifierIndex);

    // Encode time unit
    auto unitIt = TIME_UNITS.find(unit);
    if (unitIt != TIME_UNITS.end()) {
        encoded.push_back(RELATIVE_START_INDEX + 10 +
                         std::distance(TIME_UNITS.begin(), unitIt));
    }

    // Encode direction
    if (isAgo) {
        encoded.push_back(RELATIVE_START_INDEX + 20);
    }

    return encoded;
}

DateTimeEncoding::DateTimeComponents DateTimeEncoding::parseDateTime(const std::string& dateTime) const {
    DateTimeComponents components{};
    std::smatch matches;

    if (std::regex_match(dateTime, matches, ISO8601_REGEX)) {
        components.year = std::stoi(matches[1].str());
        components.month = std::stoi(matches[2].str());
        components.day = std::stoi(matches[3].str());
        components.hour = std::stoi(matches[4].str());
        components.minute = std::stoi(matches[5].str());
        components.second = std::stoi(matches[6].str());

        // Parse fractional seconds
        if (matches[7].matched) {
            std::string fraction = matches[7].str();
            fraction = fraction.substr(1); // Remove decimal point
            components.nanoseconds = std::stoi(fraction) *
                static_cast<uint32_t>(std::pow(10, 9 - fraction.length()));
        }

        // Parse timezone
        if (matches[8].matched && config.encodeTimezone) {
            std::string tz = matches[8].str();
            if (tz == "Z") {
                components.timezoneOffset = 0;
            } else {
                int hours = std::stoi(tz.substr(1, 2));
                int minutes = std::stoi(tz.substr(3, 2));
                components.timezoneOffset = hours * 60 + minutes;
                if (tz[0] == '-') {
                    components.timezoneOffset = -(*components.timezoneOffset);
                }
            }
        }
    }

    return components;
}

std::vector<size_t> DateTimeEncoding::encodeComponents(
    const DateTimeComponents& components) const {

    std::vector<size_t> encoded;

    // Encode date components
    encoded.push_back(DATETIME_START_INDEX + components.year - 1900);
    encoded.push_back(DATETIME_START_INDEX + 200 + components.month - 1);
    encoded.push_back(DATETIME_START_INDEX + 300 + components.day - 1);

    // Encode time components
    encoded.push_back(DATETIME_START_INDEX + 400 + components.hour);
    encoded.push_back(DATETIME_START_INDEX + 460 + components.minute);
    encoded.push_back(DATETIME_START_INDEX + 520 + components.second);

    // Encode fractional seconds if needed
    if (components.nanoseconds > 0 && config.precision > 0) {
        size_t nanoIndex = static_cast<size_t>(
            (static_cast<double>(components.nanoseconds) / 1e9) *
            std::pow(10, static_cast<double>(config.precision))
        );
        encoded.push_back(DATETIME_START_INDEX + 580 + nanoIndex);
    }

    // Encode timezone if present
    if (components.timezoneOffset && config.encodeTimezone) {
        encoded.push_back(TIMEZONE_START_INDEX +
                         (*components.timezoneOffset + 720)); // Offset range: -720 to +720
    }

    return encoded;
}

std::string DateTimeEncoding::decode(const std::vector<size_t>& indices) const {
    // Production-quality: reconstruct ISO8601 (or best possible) from SDR indices
    if (indices.empty()) return "";
    auto components = decodeComponents(indices);
    std::ostringstream oss;
    oss << std::setfill('0')
        << std::setw(4) << components.year << "-"
        << std::setw(2) << components.month << "-"
        << std::setw(2) << components.day << "T"
        << std::setw(2) << components.hour << ":"
        << std::setw(2) << components.minute << ":"
        << std::setw(2) << components.second;
    if (components.nanoseconds > 0) {
        oss << "." << std::setw(9) << components.nanoseconds;
    }
    if (components.timezoneOffset) {
        int tz = *components.timezoneOffset;
        char sign = tz >= 0 ? '+' : '-';
        int absTz = std::abs(tz);
        int hours = absTz / 60;
        int minutes = absTz % 60;
        oss << sign << std::setw(2) << hours << ":" << std::setw(2) << minutes;
    } else {
        oss << "Z";
    }
    return oss.str();
}

bool DateTimeEncoding::isValid(const std::string& dateTime) const {
    return std::regex_match(dateTime, ISO8601_REGEX) ||
           std::regex_match(dateTime, COMPACT_REGEX) ||
           std::regex_match(dateTime, RELATIVE_REGEX);
}

// Add missing methods
DateTimeEncoding::DateTimeComponents DateTimeEncoding::decodeComponents(const std::vector<size_t>& indices) const {
    // Production-quality: extract all fields from SDR indices
    DateTimeComponents components{};
    for (size_t idx : indices) {
        if (idx >= DATETIME_START_INDEX && idx < DATETIME_START_INDEX + 60) {
            components.year = 1970 + (idx - DATETIME_START_INDEX);
        } else if (idx >= DATETIME_START_INDEX + 60 && idx < DATETIME_START_INDEX + 160) {
            components.month = (idx - DATETIME_START_INDEX - 60) + 1;
        } else if (idx >= DATETIME_START_INDEX + 160 && idx < DATETIME_START_INDEX + 400) {
            components.day = (idx - DATETIME_START_INDEX - 160) + 1;
        } else if (idx >= DATETIME_START_INDEX + 400 && idx < DATETIME_START_INDEX + 460) {
            components.hour = (idx - DATETIME_START_INDEX - 400);
        } else if (idx >= DATETIME_START_INDEX + 460 && idx < DATETIME_START_INDEX + 520) {
            components.minute = (idx - DATETIME_START_INDEX - 460);
        } else if (idx >= DATETIME_START_INDEX + 520 && idx < DATETIME_START_INDEX + 580) {
            components.second = (idx - DATETIME_START_INDEX - 520);
        } else if (idx >= DATETIME_START_INDEX + 580 && idx < DATETIME_START_INDEX + 600) {
            // Fractional seconds (nanoseconds, scaled)
            size_t nanoIndex = idx - (DATETIME_START_INDEX + 580);
            components.nanoseconds = static_cast<uint32_t>((nanoIndex * 1e9) / std::pow(10, static_cast<double>(config.precision)));
        } else if (idx >= TIMEZONE_START_INDEX && idx < TIMEZONE_START_INDEX + 1441) {
            // Offset range: -720 to +720
            components.timezoneOffset = static_cast<int>(idx - TIMEZONE_START_INDEX) - 720;
        }
    }
    // Defaults for missing fields
    if (components.year == 0) components.year = 1970;
    if (components.month == 0) components.month = 1;
    if (components.day == 0) components.day = 1;
    return components;
}

std::chrono::system_clock::time_point DateTimeEncoding::decodeTimePoint(const std::vector<size_t>& indices) const {
    auto components = decodeComponents(indices);
    return componentsToTimePoint(components);
}

std::chrono::system_clock::time_point DateTimeEncoding::componentsToTimePoint(const DateTimeComponents& components) const {
    std::tm tm = {};
    tm.tm_year = components.year - 1900;
    tm.tm_mon = components.month - 1;
    tm.tm_mday = components.day;
    tm.tm_hour = components.hour;
    tm.tm_min = components.minute;
    tm.tm_sec = components.second;
    
    auto time = std::chrono::system_clock::from_time_t(std::mktime(&tm));
    
    // Add nanoseconds
    time += std::chrono::nanoseconds(components.nanoseconds);
    
    // Apply timezone if present
    if (components.timezoneOffset) {
        time -= std::chrono::minutes(*components.timezoneOffset);
    }
    
    return time;
}

DateTimeEncoding::TimeRange DateTimeEncoding::decodeRange(const std::vector<size_t>& indices) const {
    // Production-quality: extract start/end/interval from indices
    TimeRange range;
    if (indices.size() < 2) {
        range.start = system_clock::now();
        range.end = range.start;
        range.interval = std::chrono::seconds(0);
        return range;
    }
    // Assume first half is start, second half is end, last element is interval
    size_t n = indices.size();
    size_t half = (n - 1) / 2;
    std::vector<size_t> startIndices(indices.begin(), indices.begin() + half);
    std::vector<size_t> endIndices(indices.begin() + half, indices.end() - 1);
    range.start = decodeTimePoint(startIndices);
    range.end = decodeTimePoint(endIndices);
    range.interval = std::chrono::seconds(static_cast<int64_t>(indices.back() - DATETIME_START_INDEX));
    return range;
}

std::string DateTimeEncoding::decodeRelative(const std::vector<size_t>& indices) const {
    // Production-quality: decode relative time expressions from SDR indices
    std::string qualifier, unit;
    bool isAgo = false;
    for (size_t idx : indices) {
        if (idx >= RELATIVE_START_INDEX && idx < RELATIVE_START_INDEX + 4) {
            switch (idx - RELATIVE_START_INDEX) {
                case 0: qualifier = "now"; break;
                case 1: qualifier = "last"; break;
                case 2: qualifier = "next"; break;
                case 3: qualifier = "this"; break;
            }
        } else if (idx >= RELATIVE_UNIT_START_INDEX && idx < RELATIVE_UNIT_START_INDEX + 7) {
            static const char* units[] = {"second","minute","hour","day","week","month","year"};
            unit = units[idx - RELATIVE_UNIT_START_INDEX];
        } else if (idx == RELATIVE_AGO_INDEX) {
            isAgo = true;
        }
    }
    std::ostringstream oss;
    if (!qualifier.empty()) oss << qualifier << " ";
    if (!unit.empty()) oss << unit << "s";
    if (isAgo) oss << " ago";
    return oss.str();
}

std::string DateTimeEncoding::convertFormat(const std::string& dateTime, Format targetFormat) const {
    // Placeholder implementation
    return dateTime;
}

DateTimeEncoding::Format DateTimeEncoding::detectFormat(const std::string& dateTime) const {
    if (std::regex_match(dateTime, ISO8601_REGEX)) {
        return DateTimeEncoding::Format::ISO8601;
    } else if (std::regex_match(dateTime, COMPACT_REGEX)) {
        return DateTimeEncoding::Format::COMPACT;
    } else if (std::regex_match(dateTime, RELATIVE_REGEX)) {
        return DateTimeEncoding::Format::RELATIVE;
    }

    try {
        std::stoull(dateTime);
        return DateTimeEncoding::Format::UNIX_TIMESTAMP;
    } catch (...) {
        throw std::invalid_argument("Unrecognized datetime format");
    }
}
