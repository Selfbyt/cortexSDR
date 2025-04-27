# CortexSDR CLI Tool Usage

This document describes how to use the `cortexsdr_cli` command-line tool provided with the CortexSDR project. This tool allows you to compress files into the `.sdr` format and decompress `.sdr` files back into their original (lossy for text) format using the CortexSDR library.

## Building the Tool

The CLI tool is built along with the main project when using the standard build script:

```bash
./build.sh
```

The executable will be located at `build/cortexsdr_cli`.

## Usage

The tool operates in two modes: compression (`-c`) and decompression (`-d`).

### Compression

To compress a file into the `.sdr` format:

```bash
build/cortexsdr_cli -c <input_file> <output_file.sdr>
```

-   `<input_file>`: The path to the file you want to compress.
-   `<output_file.sdr>`: The desired path for the output `.sdr` file.

**Example:**

```bash
build/cortexsdr_cli -c README.md compressed_readme.sdr
```

This will read `README.md`, compress its content using the CortexSDR library (currently treating it as text), and save the compressed representation to `compressed_readme.sdr`. The tool will output the original size, the size of the compressed data, and the resulting compression ratio.

### Decompression

To decompress an `.sdr` file:

```bash
build/cortexsdr_cli -d <input_file.sdr> <output_file>
```

-   `<input_file.sdr>`: The path to the `.sdr` file you want to decompress.
-   `<output_file>`: The desired path for the decompressed output file.

**Example:**

```bash
build/cortexsdr_cli -d compressed_readme.sdr decompressed_readme.txt
```

This will read the compressed data from `compressed_readme.sdr`, use the CortexSDR library to decode it, and write the resulting content to `decompressed_readme.txt`.

### Important Note on Text Decompression

The current text encoding/decoding mechanism in the CortexSDR library is **lossy**. This means that when compressing and then decompressing a text file, the resulting output file **will not be identical** to the original. Information such as original capitalization, exact whitespace, punctuation positioning, and word order might be lost or altered during the process.

The library's primary strength lies in creating semantic representations for pattern matching and similarity analysis, not in perfect lossless reconstruction of text files.
