# Cortex SDR (Sparse Distributed Representation)

A C++ implementation of Sparse Distributed Representations for efficient encoding and storage of various data types including text, numbers, dates, and special characters.

## Overview

This library implements a Sparse Distributed Representation system that can encode different types of data into a binary vector format where only a small subset of bits are active. This approach provides several benefits:

- Efficient storage through sparse representation
- Noise-resistant data encoding
- Pattern matching capabilities
- Semantic similarity preservation
- Flexible combination of different data types

## Components

### Core SDR Class (`cortexSDR.cpp`)

The main class `SparseDistributedRepresentation` handles:
- Vector management (2000-bit vectors)
- Encoding/decoding coordination
- Data type separation
- Memory optimization through sparse storage

### Encoders

#### 1. Word Encoding (`WordEncoding.hpp/cpp`)
- Handles text-to-SDR conversion
- Vocabulary-based encoding
- Maps words to specific bit positions
- Maintains word-to-index mapping for efficient lookup

#### 2. Number Encoding (`NumberEncoding.hpp/cpp`)
- Converts numerical values to SDR
- Uses bucket-based discretization
- Supports ranges from -1000 to 1000 by default
- Implements smooth transitions between adjacent values

#### 3. Special Character Encoding (`SpecialCharacterEncoding.hpp/cpp`)
- Encodes non-alphanumeric characters
- Maps characters to specific bit ranges
- Preserves punctuation and symbols

#### 4. DateTime Encoding (`DateTimeEncoding.hpp/cpp`)
- Handles temporal data
- Converts dates and times to SDR format
- Preserves temporal relationships

## Usage

### Basic Example
