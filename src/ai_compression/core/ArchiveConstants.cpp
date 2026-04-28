/**
 * @file ArchiveConstants.cpp
 * @brief Defines archive magic and version used across compression/decompression.
 */
#include "ArchiveConstants.hpp"

namespace CortexAICompression {

// Define simple magic number and version for the archive format
const char ARCHIVE_MAGIC[8] = {'C', 'O', 'R', 'T', 'E', 'X', 'S', 'R'};
const uint32_t ARCHIVE_VERSION = 2;

} // namespace CortexAICompression
