/**
 * @file ArchiveConstants.hpp
 * @brief Constants defining the archive format magic and version.
 */
#ifndef ARCHIVE_CONSTANTS_HPP
#define ARCHIVE_CONSTANTS_HPP

#include <cstdint>

namespace CortexAICompression {

/** Magic number for identifying Cortex AI archives. */
extern const char ARCHIVE_MAGIC[8];
/** Version of the archive format for compatibility checks. */
extern const uint32_t ARCHIVE_VERSION;

} // namespace CortexAICompression

#endif // ARCHIVE_CONSTANTS_HPP
