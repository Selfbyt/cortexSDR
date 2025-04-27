#ifndef DATE_TIME_ENCODING_HPP
#define DATE_TIME_ENCODING_HPP

#include <chrono>
#include <string>
#include <vector>
#include <cstdint>
#include <optional>

class DateTimeEncoding {
public:
    struct TimeRange {
        std::chrono::system_clock::time_point start;
        std::chrono::system_clock::time_point end;
        std::chrono::system_clock::duration interval;
    };

    enum class Format {
        ISO8601,        // "2024-02-20T15:30:45Z"
        UNIX_TIMESTAMP, // 1708442445
        COMPACT,        // 20240220153045
        RELATIVE       // "2 days ago", "next week"
    };

    struct EncodingConfig {
        Format preferredFormat = Format::ISO8601;
        bool enableCompression = true;
        bool encodeTimezone = true;
        uint8_t precision = 6; // Decimal places for fractional seconds
    };

    explicit DateTimeEncoding(const EncodingConfig& config);

    // Encoding methods
    std::vector<size_t> encode(const std::string& dateTime) const;
    std::vector<size_t> encode(const std::chrono::system_clock::time_point& timePoint) const;
    std::vector<size_t> encodeRange(const TimeRange& range) const;
    std::vector<size_t> encodeRelative(const std::string& relativeTime) const;

    // Decoding methods
    std::string decode(const std::vector<size_t>& indices) const;
    std::chrono::system_clock::time_point decodeTimePoint(const std::vector<size_t>& indices) const;
    TimeRange decodeRange(const std::vector<size_t>& indices) const;
    std::string decodeRelative(const std::vector<size_t>& indices) const;

    // Utility methods
    bool isValid(const std::string& dateTime) const;
    Format detectFormat(const std::string& dateTime) const;
    std::string convertFormat(const std::string& dateTime, Format targetFormat) const;

private:
    EncodingConfig config;
    static constexpr size_t DATETIME_START_INDEX = 2000;
    static constexpr size_t TIMEZONE_START_INDEX = 2500;
    static constexpr size_t RELATIVE_START_INDEX = 2750;

    // Internal encoding structures
    struct DateTimeComponents {
        uint16_t year;
        uint8_t month;
        uint8_t day;
        uint8_t hour;
        uint8_t minute;
        uint8_t second;
        uint32_t nanoseconds;
        std::optional<int32_t> timezoneOffset; // in minutes
    };

    // Helper methods
    DateTimeComponents parseDateTime(const std::string& dateTime) const;
    std::vector<size_t> encodeComponents(const DateTimeComponents& components) const;
    DateTimeComponents decodeComponents(const std::vector<size_t>& indices) const;
    std::chrono::system_clock::time_point componentsToTimePoint(const DateTimeComponents& components) const;
};

#endif // DATE_TIME_ENCODING_HPP
