#!/bin/bash

# Setup script for Homebrew tap
# This script helps create and configure the Homebrew tap for cortexsdr

set -e

echo "Setting up Homebrew tap for cortexsdr..."

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: Please run this script from the cortexsDR root directory"
    exit 1
fi

# Create Formula directory
mkdir -p Formula

# Create the Homebrew formula
cat > Formula/cortexsdr.rb << 'EOF'
class Cortexsdr < Formula
  desc "CortexSDR AI compression CLI tool"
  homepage "https://github.com/Selfbyt/cortexSDR"
  url "https://github.com/Selfbyt/cortexSDR/releases/download/v1.0.0/cortexsdr-cpp-sdk-linux.tar.gz"
  sha256 "PLACEHOLDER_SHA256"
  version "1.0.0"
  license "MIT"

  depends_on "cmake" => :build
  depends_on "protobuf"
  depends_on "zlib"
  depends_on "nlohmann-json"

  def install
    # Extract the SDK
    system "tar", "-xzf", "cortexsdr-cpp-sdk-linux.tar.gz"
    
    # Install binaries
    bin.install "sdk_release/bin/cortexsdr_ai_compression_cli" => "cortexsdr"
    bin.install "sdk_release/bin/cortexsdr_cli" => "cortexsdr-cli"
    
    # Install libraries
    lib.install Dir["sdk_release/lib/*"]
    
    # Install headers
    include.install Dir["sdk_release/include/*"]
    
    # Install examples
    share.install "sdk_release/examples" => "cortexsdr"
    
    # Install documentation
    doc.install "sdk_release/docs"
  end

  test do
    system "#{bin}/cortexsdr", "--help"
  end
end
EOF

echo "Homebrew formula created at Formula/cortexsdr.rb"
echo ""
echo "Next steps:"
echo "1. Create a new repository on GitHub called 'homebrew-cortexsdr'"
echo "   - Go to https://github.com/new"
echo "   - Repository name: homebrew-cortexsdr"
echo "   - Make it public"
echo "   - Don't initialize with README (we'll push our own)"
echo ""
echo "2. Clone the new repository:"
echo "   git clone https://github.com/Selfbyt/homebrew-cortexsdr.git"
echo "   cd homebrew-cortexsdr"
echo ""
echo "3. Copy the formula:"
echo "   cp ../cortexSDR/Formula/cortexsdr.rb Formula/"
echo ""
echo "4. Commit and push:"
echo "   git add Formula/cortexsdr.rb"
echo "   git commit -m 'Add cortexsdr formula'"
echo "   git push origin main"
echo ""
echo "5. Update the formula with the correct SHA256:"
echo "   - Download the release tarball"
echo "   - Run: shasum -a 256 cortexsdr-cpp-sdk-linux.tar.gz"
echo "   - Update the sha256 in Formula/cortexsdr.rb"
echo ""
echo "6. Users can then install with:"
echo "   brew tap Selfbyt/cortexsdr"
echo "   brew install cortexsdr"
echo ""
echo "The workflow will automatically update this formula on each release." 