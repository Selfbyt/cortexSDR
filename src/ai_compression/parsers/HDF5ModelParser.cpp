/**
 * @file HDF5ModelParser.cpp
 * @brief Implementation of HDF5/Keras model parsing into archive segments.
 */
#include "HDF5ModelParser.hpp"
#include <stdexcept>
#include <regex>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <fstream>

#ifdef ENABLE_HDF5
#include <hdf5.h>
#include <hdf5_hl.h>
#endif

namespace CortexAICompression {

HDF5ModelParser::HDF5ModelParser() {
#ifdef ENABLE_HDF5
    // Initialize HDF5 library
    H5open();
#endif
}

HDF5ModelParser::~HDF5ModelParser() {
#ifdef ENABLE_HDF5
    // Close HDF5 library
    H5close();
#endif
}

#ifdef ENABLE_HDF5

SegmentType HDF5ModelParser::hdf5DataTypeToSegmentType(hid_t data_type) const {
    H5T_class_t type_class = H5Tget_class(data_type);
    size_t type_size = H5Tget_size(data_type);
    
    if (type_class == H5T_FLOAT) {
        if (type_size == 4) {
            return SegmentType::WEIGHTS_FP32;
        } else if (type_size == 2) {
            return SegmentType::WEIGHTS_FP16;
        }
    } else if (type_class == H5T_INTEGER) {
        if (type_size == 1) {
            return SegmentType::WEIGHTS_INT8;
        } else if (type_size == 4) {
            return SegmentType::WEIGHTS_FP32; // Placeholder
        }
    }
    
    return SegmentType::UNKNOWN;
}

TensorMetadata HDF5ModelParser::extractTensorMetadata(const HDF5DatasetInfo& datasetInfo) const {
    TensorMetadata metadata;
    
    // Convert HDF5 dimensions to our format
    for (hsize_t dim : datasetInfo.dimensions) {
        metadata.dimensions.push_back(static_cast<size_t>(dim));
    }
    
    // Calculate sparsity (placeholder - would need to analyze actual data)
    metadata.sparsity_ratio = 0.0f;
    metadata.is_sorted = false;
    
    return metadata;
}

std::string HDF5ModelParser::extractLayerName(const std::string& datasetName) const {
    // HDF5 dataset names often follow patterns like:
    // "layer_1/weight", "dense_2/bias", "conv2d_3/kernel"
    std::regex layer_pattern(R"(([^/]+)/)");
    std::smatch matches;
    if (std::regex_search(datasetName, matches, layer_pattern)) {
        return matches[1].str();
    }
    return datasetName;
}

size_t HDF5ModelParser::extractLayerIndex(const std::string& datasetName) const {
    // Extract numeric index from dataset names like "layer_1", "dense_2", "conv2d_3"
    std::regex index_pattern(R"((\d+))");
    std::smatch matches;
    if (std::regex_search(datasetName, matches, index_pattern)) {
        try {
            return std::stoul(matches[1].str());
        } catch (const std::exception& e) {
            std::cerr << "Warning: Invalid layer index in dataset name: " << datasetName << std::endl;
        }
    }
    return 0;
}

std::vector<std::byte> HDF5ModelParser::readDatasetData(hid_t file_id, const std::string& datasetName, const std::vector<hsize_t>& dimensions, hid_t data_type) const {
    std::vector<std::byte> data;
    
    try {
        // Open the dataset
        hid_t dataset_id = H5Dopen2(file_id, datasetName.c_str(), H5P_DEFAULT);
        if (dataset_id < 0) {
            throw std::runtime_error("Failed to open dataset: " + datasetName);
        }
        
        // Get the dataspace
        hid_t space_id = H5Dget_space(dataset_id);
        if (space_id < 0) {
            H5Dclose(dataset_id);
            throw std::runtime_error("Failed to get dataspace for dataset: " + datasetName);
        }
        
        // Calculate total size
        size_t type_size = H5Tget_size(data_type);
        size_t total_elements = 1;
        for (hsize_t dim : dimensions) {
            total_elements *= static_cast<size_t>(dim);
        }
        size_t total_size = total_elements * type_size;
        
        // Allocate buffer
        data.resize(total_size);
        
        // Read the data
        herr_t status = H5Dread(dataset_id, data_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
        if (status < 0) {
            H5Sclose(space_id);
            H5Dclose(dataset_id);
            throw std::runtime_error("Failed to read data from dataset: " + datasetName);
        }
        
        // Clean up
        H5Sclose(space_id);
        H5Dclose(dataset_id);
        
    } catch (const std::exception& e) {
        std::cerr << "Error reading HDF5 dataset data: " << e.what() << std::endl;
        // Return empty data on error
        data.clear();
    }
    
    return data;
}

void HDF5ModelParser::traverseGroup(hid_t group_id, const std::string& groupName, std::vector<HDF5DatasetInfo>& datasets) const {
    hsize_t num_objects;
    herr_t status = H5Gget_num_objs(group_id, &num_objects);
    if (status < 0) {
        return;
    }
    
    for (hsize_t i = 0; i < num_objects; ++i) {
        char obj_name[256];
        ssize_t name_len = H5Gget_objname_by_idx(group_id, i, obj_name, sizeof(obj_name));
        if (name_len < 0) {
            continue;
        }
        
        std::string full_name = groupName.empty() ? obj_name : groupName + "/" + obj_name;
        
        // Get object type
        H5G_obj_t obj_type = H5Gget_objtype_by_idx(group_id, i);
        
        if (obj_type == H5G_DATASET) {
            // It's a dataset, extract information
            hid_t dataset_id = H5Dopen2(group_id, obj_name, H5P_DEFAULT);
            if (dataset_id >= 0) {
                HDF5DatasetInfo datasetInfo;
                datasetInfo.name = full_name;
                
                // Get dataspace
                hid_t space_id = H5Dget_space(dataset_id);
                if (space_id >= 0) {
                    int ndims = H5Sget_simple_extent_ndims(space_id);
                    datasetInfo.dimensions.resize(ndims);
                    H5Sget_simple_extent_dims(space_id, datasetInfo.dimensions.data(), nullptr);
                    H5Sclose(space_id);
                }
                
                // Get data type
                datasetInfo.data_type = H5Dget_type(dataset_id);
                
                // Calculate size
                size_t type_size = H5Tget_size(datasetInfo.data_type);
                size_t total_elements = 1;
                for (hsize_t dim : datasetInfo.dimensions) {
                    total_elements *= static_cast<size_t>(dim);
                }
                datasetInfo.size_bytes = total_elements * type_size;
                
                // Read data
                datasetInfo.data = readDatasetData(group_id, obj_name, datasetInfo.dimensions, datasetInfo.data_type);
                
                datasets.push_back(std::move(datasetInfo));
                H5Dclose(dataset_id);
            }
        } else if (obj_type == H5G_GROUP) {
            // It's a group, traverse recursively
            hid_t sub_group_id = H5Gopen2(group_id, obj_name, H5P_DEFAULT);
            if (sub_group_id >= 0) {
                traverseGroup(sub_group_id, full_name, datasets);
                H5Gclose(sub_group_id);
            }
        }
    }
}

std::vector<HDF5ModelParser::HDF5DatasetInfo> HDF5ModelParser::extractDatasetInfo(const std::string& modelPath) const {
    std::vector<HDF5DatasetInfo> datasets;
    
    try {
        
        // Open the HDF5 file
        hid_t file_id = H5Fopen(modelPath.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id < 0) {
            throw ParsingError("Failed to open HDF5 file: " + modelPath);
        }
        
        // Traverse the file structure
        traverseGroup(file_id, "", datasets);
        
        // Close the file
        H5Fclose(file_id);
        
        
    } catch (const std::exception& e) {
        std::cerr << "Error extracting HDF5 dataset info: " << e.what() << std::endl;
        throw ParsingError("Failed to extract HDF5 dataset information: " + std::string(e.what()));
    }
    
    return datasets;
}

ModelSegment HDF5ModelParser::createSegmentFromDataset(const HDF5DatasetInfo& datasetInfo) const {
    ModelSegment segment;
    segment.name = datasetInfo.name;
    segment.type = hdf5DataTypeToSegmentType(datasetInfo.data_type);
    segment.data = datasetInfo.data;
    segment.original_size = datasetInfo.size_bytes;
    segment.tensor_metadata = extractTensorMetadata(datasetInfo);
    segment.layer_name = extractLayerName(datasetInfo.name);
    segment.layer_index = extractLayerIndex(datasetInfo.name);
    
    // Determine layer type based on dataset name
    if (datasetInfo.name.find("weight") != std::string::npos || datasetInfo.name.find("kernel") != std::string::npos) {
        segment.layer_type = "WEIGHTS";
    } else if (datasetInfo.name.find("bias") != std::string::npos) {
        segment.layer_type = "BIAS";
    } else if (datasetInfo.name.find("gamma") != std::string::npos) {
        segment.layer_type = "BATCH_NORM_GAMMA";
    } else if (datasetInfo.name.find("beta") != std::string::npos) {
        segment.layer_type = "BATCH_NORM_BETA";
    } else if (datasetInfo.name.find("moving_mean") != std::string::npos) {
        segment.layer_type = "BATCH_NORM_MEAN";
    } else if (datasetInfo.name.find("moving_variance") != std::string::npos) {
        segment.layer_type = "BATCH_NORM_VAR";
    } else {
        segment.layer_type = "UNKNOWN";
    }
    
    return segment;
}

#endif // ENABLE_HDF5

std::vector<ModelSegment> HDF5ModelParser::parse(const std::string& modelPath) const {
    std::vector<ModelSegment> segments;
    
#ifdef ENABLE_HDF5
    try {
        
        auto datasetInfos = extractDatasetInfo(modelPath);
        segments.reserve(datasetInfos.size());
        
        for (const auto& datasetInfo : datasetInfos) {
            segments.push_back(createSegmentFromDataset(datasetInfo));
        }
        
        
    } catch (const std::exception& e) {
        std::cerr << "Error parsing HDF5 model: " << e.what() << std::endl;
        throw ParsingError("Failed to parse HDF5 model: " + std::string(e.what()));
    }
#else
    throw std::runtime_error("HDF5 model support is disabled. Please enable ENABLE_HDF5 to use this feature.");
#endif
    
    return segments;
}

std::vector<ModelSegment> HDF5ModelParser::parseWithChunking(const std::string& modelPath) const {
    auto segments = parse(modelPath);
    
    // Group segments by layer for better compression
    std::map<size_t, std::vector<ModelSegment*>> layerGroups;
    for (auto& segment : segments) {
        layerGroups[segment.layer_index].push_back(&segment);
    }
    
    // Sort segments within each layer by type
    for (auto& [layer, group] : layerGroups) {
        std::sort(group.begin(), group.end(),
                 [](const ModelSegment* a, const ModelSegment* b) {
                     return static_cast<int>(a->type) < static_cast<int>(b->type);
                 });
    }
    
    // Reorder segments for optimal compression
    std::vector<ModelSegment> reorderedSegments;
    reorderedSegments.reserve(segments.size());
    
    // First, add non-layer segments (global params, etc.)
    for (const auto& segment : segments) {
        if (segment.layer_index == 0 && segment.layer_name.empty()) {
            reorderedSegments.push_back(segment);
        }
    }
    
    // Then add layer segments in order
    for (const auto& [layer, group] : layerGroups) {
        if (layer > 0 || !group.front()->layer_name.empty()) {
            for (const auto* segment : group) {
                reorderedSegments.push_back(*segment);
            }
        }
    }
    
    return reorderedSegments;
}

} // namespace CortexAICompression
