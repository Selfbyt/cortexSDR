#ifndef ARCHIVE_CONSTANTS_HPP
#define ARCHIVE_CONSTANTS_HPP

#include <cstdint>

namespace CortexAICompression {

// Define simple magic number and version for the archive format
extern const char ARCHIVE_MAGIC[8];
extern const uint32_t ARCHIVE_VERSION;

} // namespace CortexAICompression

#endif // ARCHIVE_CONSTANTS_HPP
