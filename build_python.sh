#!/bin/bash

# Build script for cortexSDR Python wrapper
echo "Building cortexSDR Python wrapper..."

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "Error: pip is not installed. Please install Python and pip first."
    exit 1
fi

# Check if pybind11 is installed
if ! pip show pybind11 &> /dev/null; then
    echo "Installing pybind11..."
    pip install pybind11
fi

# Create build directory if it doesn't exist
mkdir -p build_python
cd build_python

# Configure with Python wrapper options
cmake -DBUILD_LIBRARY=ON -DBUILD_PYTHON_WRAPPER=ON -DBUILD_DESKTOP=OFF -DBUILD_FIRMWARE=OFF ..

# Build the Python wrapper
make -j$(nproc)

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Python wrapper build successful!"
    echo ""
    echo "To install the Python module, run:"
    echo "sudo make install"
    echo ""
    echo "To use in your Python code:"
    echo "import cortexsdr"
    echo "sdr = cortexsdr.SDR(['word1', 'word2'])"
    echo "encoded = sdr.encode_text('your text here')"
else
    echo "Python wrapper build failed!"
    exit 1
fi

echo "Build complete!"
