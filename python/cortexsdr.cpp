#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../src/cortexSDR.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cortexsdr, m) {
    m.doc() = "Python bindings for CortexSDR - Sparse Distributed Representation library";
    
    // Wrap the SparseDistributedRepresentation class
    py::class_<SparseDistributedRepresentation>(m, "SDR")
        .def(py::init<>())
        .def(py::init<const std::vector<std::string>&>())
        .def("encode_text", &SparseDistributedRepresentation::encodeText)
        .def("encode_number", &SparseDistributedRepresentation::encodeNumber)
        .def("encode_date_time", &SparseDistributedRepresentation::encodeDateTime)
        .def("decode", &SparseDistributedRepresentation::decode)
        .def("get_active_bits", [](const SparseDistributedRepresentation& self) {
            // Return the active bit positions as a Python list
            return self.getActiveBits();
        })
        .def("similarity", &SparseDistributedRepresentation::similarity)
        .def("merge", &SparseDistributedRepresentation::merge)
        .def("clear", &SparseDistributedRepresentation::clear)
        .def_property_readonly("size", &SparseDistributedRepresentation::size)
        .def_property_readonly("active_bit_count", &SparseDistributedRepresentation::activeBitCount)
        .def("__repr__", [](const SparseDistributedRepresentation& self) {
            return "SDR(active_bits=" + std::to_string(self.activeBitCount()) + 
                   ", size=" + std::to_string(self.size()) + ")";
        });
    
    // Add version information
    m.attr("__version__") = "1.0.0";
}
