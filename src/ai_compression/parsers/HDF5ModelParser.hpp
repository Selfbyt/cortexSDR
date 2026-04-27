/**
 * @file HDF5ModelParser.hpp
 * @brief Parser for HDF5/Keras model format (.h5 files)
 */
#ifndef HDF5_MODEL_PARSER_HPP
#define HDF5_MODEL_PARSER_HPP

#include "../core/AIModelParser.hpp"
#include "../core/ModelSegment.hpp"
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <fstream>

#ifdef ENABLE_HDF5
#include <hdf5.h>
#endif

namespace CortexAICompression {

/**
 * @brief Parse HDF5/Keras model format into compression-ready segments.
 */
class HDF5ModelParser : public IAIModelParser {
public:
    HDF5ModelParser();
    ~HDF5ModelParser() override;

    std::vector<ModelSegment> parse(const std::string& modelPath) const override;
    std::vector<ModelSegment> parseWithChunking(const std::string& modelPath) const override;

private:
#ifdef ENABLE_HDF5
    // Helper struct for HDF5 dataset info
    struct HDF5DatasetInfo {
        std::string name;
        std::vector<hsize_t> dimensions;
        hid_t data_type;
        size_t size_bytes;
        std::vector<std::byte> data;
    };

    // Helper methods
    std::vector<HDF5DatasetInfo> extractDatasetInfo(const std::string& modelPath) const;
    ModelSegment createSegmentFromDataset(const HDF5DatasetInfo& datasetInfo) const;
    SegmentType hdf5DataTypeToSegmentType(hid_t data_type) const;
    TensorMetadata extractTensorMetadata(const HDF5DatasetInfo& datasetInfo) const;
    std::string extractLayerName(const std::string& datasetName) const;
    size_t extractLayerIndex(const std::string& datasetName) const;
    std::vector<std::byte> readDatasetData(hid_t file_id, const std::string& datasetName, const std::vector<hsize_t>& dimensions, hid_t data_type) const;
    void traverseGroup(hid_t group_id, const std::string& groupName, std::vector<HDF5DatasetInfo>& datasets) const;
#endif
};

} // namespace CortexAICompression

#endif // HDF5_MODEL_PARSER_HPP
