#ifndef ARCHIVE_CONSTANTS_HPP
#define ARCHIVE_CONSTANTS_HPP

#include <cstdint>

namespace CortexAICompression {

// Define simple magic number and version for the archive format
extern const char ARCHIVE_MAGIC[4];
extern const uint16_t ARCHIVE_VERSION;

} // namespace CortexAICompression

#endif // ARCHIVE_CONSTANTS_HPP
