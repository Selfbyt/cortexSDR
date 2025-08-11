#include "ModelManager.h"
#include <QFileDialog>
#include <QStandardPaths>
#include <QDataStream>
#include <QFileInfo>
#include <QProcess>
#include <QCoreApplication>
#include <chrono>
#include <algorithm>

ModelManager::ModelManager(QObject *parent)
    : QObject(parent)
    , m_currentModelPath("")
    , m_currentModelFormat("")
    , m_modelLoaded(false)
    , m_compressorHandle(nullptr)
    , m_inferenceHandle(nullptr)
{
    // Connect watchers
    connect(&m_compressionWatcher, &QFutureWatcher<CompressionResult>::finished, 
            this, &ModelManager::onCompressionFinished);
    connect(&m_inferenceWatcher, &QFutureWatcher<InferenceResult>::finished, 
            this, &ModelManager::onInferenceFinished);
    connect(&m_benchmarkWatcher, &QFutureWatcher<BenchmarkResult>::finished, 
            this, &ModelManager::onBenchmarkFinished);
}

ModelManager::~ModelManager()
{
    // Clean up SDK handles
    if (m_compressorHandle) {
        cortex_compressor_free(m_compressorHandle);
    }
    if (m_inferenceHandle) {
        cortex_inference_engine_free(m_inferenceHandle);
    }
}

bool ModelManager::loadModel(const QString &modelPath, const QString &format)
{
    QMutexLocker locker(&m_mutex);
    
    try {
        // Detect format if not specified
        QString actualFormat = format;
        if (format == "auto" || format.isEmpty()) {
            actualFormat = detectModelFormat(modelPath);
        }
        
        // Initialize compression options
        CortexCompressionOptions options;
        CortexError error = cortex_compression_options_init(&options);
        if (error.code != 0) {
            emit modelLoadFailed(getErrorMessage(error));
            return false;
        }
        
        // Create compressor
        error = cortex_compressor_create(
            modelPath.toUtf8().constData(),
            actualFormat.toUtf8().constData(),
            &options,
            &m_compressorHandle
        );
        
        if (error.code != 0) {
            emit modelLoadFailed(getErrorMessage(error));
            return false;
        }
        
        // Update state
        m_currentModelPath = modelPath;
        m_currentModelFormat = actualFormat;
        m_modelLoaded = true;
        
        emit modelLoaded(modelPath);
        return true;
        
    } catch (const std::exception &e) {
        emit modelLoadFailed(QString("Exception: %1").arg(e.what()));
        return false;
    }
}

bool ModelManager::loadModelFromURL(const QString &url)
{
    // This would download the model from URL and then load it
    // For now, just emit an error
    emit modelLoadFailed("URL loading not implemented yet");
    return false;
}

void ModelManager::compressModel(const QString &outputPath, float sparsity, 
                                int compressionLevel, bool useQuantization, int quantizationBits)
{
    if (!m_modelLoaded || !m_compressorHandle) {
        emit error("No model loaded");
        return;
    }
    
    emit compressionStarted();
    
    // Run compression in background thread
    QFuture<CompressionResult> future = QtConcurrent::run(
        [this, outputPath, sparsity, compressionLevel, useQuantization, quantizationBits]() {
            return performCompression(outputPath, sparsity, compressionLevel, useQuantization, quantizationBits);
        }
    );
    
    m_compressionWatcher.setFuture(future);
}

void ModelManager::runTextInference(const QString &inputText, int maxLength)
{
    if (!m_modelLoaded) {
        emit error("No model loaded");
        return;
    }
    
    emit inferenceStarted();
    
    // Run inference in background thread
    QFuture<InferenceResult> future = QtConcurrent::run(
        [this, inputText, maxLength]() {
            return performTextInference(inputText, maxLength);
        }
    );
    
    m_inferenceWatcher.setFuture(future);
}

void ModelManager::runAudioInference(const QString &audioPath)
{
    if (!m_modelLoaded) {
        emit error("No model loaded");
        return;
    }
    
    emit inferenceStarted();
    
    // Run inference in background thread
    QFuture<InferenceResult> future = QtConcurrent::run(
        [this, audioPath]() {
            return performAudioInference(audioPath);
        }
    );
    
    m_inferenceWatcher.setFuture(future);
}

void ModelManager::runBenchmark(int numRuns)
{
    if (!m_modelLoaded) {
        emit error("No model loaded");
        return;
    }
    
    emit benchmarkStarted();
    
    // Run benchmark in background thread
    QFuture<BenchmarkResult> future = QtConcurrent::run(
        [this, numRuns]() {
            return performBenchmark(numRuns);
        }
    );
    
    m_benchmarkWatcher.setFuture(future);
}

QString ModelManager::detectModelFormat(const QString &modelPath)
{
    QFileInfo fileInfo(modelPath);
    QString extension = fileInfo.suffix().toLower();
    
    if (extension == "onnx") return "onnx";
    if (extension == "gguf") return "gguf";
    if (extension == "pt" || extension == "pth") return "pytorch";
    if (extension == "pb" || extension == "h5") return "tensorflow";
    
    return "onnx"; // Default to ONNX
}

CompressionResult ModelManager::performCompression(const QString &outputPath, float sparsity,
                                                  int compressionLevel, bool useQuantization, int quantizationBits)
{
    CompressionResult result;
    result.modelPath = m_currentModelPath;
    result.outputPath = outputPath;
    result.success = false;
    
    try {
        // Update compression options
        CortexCompressionOptions options;
        CortexError error = cortex_compression_options_init(&options);
        if (error.code != 0) {
            result.errorMessage = getErrorMessage(error);
            return result;
        }
        
        options.sparsity = sparsity;
        options.compression_level = compressionLevel;
        options.use_quantization = useQuantization ? 1 : 0;
        options.quantization_bits = quantizationBits;
        
        // Perform compression
        auto startTime = std::chrono::high_resolution_clock::now();
        
        error = cortex_compressor_compress(m_compressorHandle, outputPath.toUtf8().constData());
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        if (error.code != 0) {
            result.errorMessage = getErrorMessage(error);
            return result;
        }
        
        // Get compression statistics
        size_t originalSize, compressedSize;
        double compressionRatio, compressionTimeMs;
        
        error = cortex_compressor_get_stats(m_compressorHandle, &originalSize, &compressedSize, 
                                          &compressionRatio, &compressionTimeMs);
        
        if (error.code == 0) {
            result.originalSize = originalSize;
            result.compressedSize = compressedSize;
            result.compressionRatio = compressionRatio;
            result.compressionTimeMs = compressionTimeMs;
        } else {
            // Use measured time if stats not available
            result.compressionTimeMs = duration.count();
        }
        
        result.success = true;
        
    } catch (const std::exception &e) {
        result.errorMessage = QString("Exception: %1").arg(e.what());
    }
    
    return result;
}

InferenceResult ModelManager::performTextInference(const QString &inputText, int maxLength)
{
    InferenceResult result;
    result.modelPath = m_currentModelPath;
    result.inputText = inputText;
    result.success = false;
    
    try {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Create inference engine if not exists
        if (!m_inferenceHandle) {
            // Try to create inference engine from compressed model
            QString compressedPath = m_currentModelPath;
            if (!compressedPath.endsWith(".sdr")) {
                // If not already compressed, we need to compress first
                compressedPath = m_currentModelPath + ".sdr";
                
                // Compress the model first
                CortexCompressionOptions options;
                cortex_compression_options_init(&options);
                options.sparsity = 0.02f;
                options.compression_level = 6;
                
                CortexError error = cortex_compressor_create(
                    m_currentModelPath.toUtf8().constData(),
                    m_currentModelFormat.toUtf8().constData(),
                    &options,
                    &m_compressorHandle
                );
                
                if (error.code != 0) {
                    result.errorMessage = QString("Failed to create compressor: %1").arg(error.message);
                    return result;
                }
                
                error = cortex_compressor_compress(m_compressorHandle, compressedPath.toUtf8().constData());
                if (error.code != 0) {
                    result.errorMessage = QString("Failed to compress model: %1").arg(error.message);
                    return result;
                }
            }
            
            // Create inference engine
            CortexError error = cortex_inference_engine_create(
                compressedPath.toUtf8().constData(),
                &m_inferenceHandle
            );
            
            if (error.code != 0) {
                result.errorMessage = QString("Failed to create inference engine: %1").arg(error.message);
                return result;
            }
        }
        
        // Convert text input to tensor (simplified tokenization)
        std::vector<float> inputTensor = textToTensor(inputText);
        
        // Prepare output buffer
        std::vector<float> outputTensor(inputTensor.size() * 2); // Estimate output size
        
        size_t actualOutputSize;
        CortexError error = cortex_inference_engine_run(
            m_inferenceHandle,
            inputTensor.data(),
            inputTensor.size(),
            outputTensor.data(),
            outputTensor.size(),
            &actualOutputSize
        );
        
        if (error.code != 0) {
            result.errorMessage = QString("Inference failed: %1").arg(error.message);
            return result;
        }
        
        // Resize output tensor to actual size
        outputTensor.resize(actualOutputSize);
        
        // Convert output tensor back to text
        result.outputText = tensorToText(outputTensor, maxLength);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        result.inferenceTimeMs = duration.count();
        
        // Get memory usage
        result.memoryUsageMB = getCurrentMemoryUsage();
        result.success = true;
        
    } catch (const std::exception &e) {
        result.errorMessage = QString("Exception: %1").arg(e.what());
    }
    
    return result;
}

InferenceResult ModelManager::performAudioInference(const QString &audioPath)
{
    InferenceResult result;
    result.modelPath = m_currentModelPath;
    result.inputText = audioPath; // Store audio path in inputText for now
    result.success = false;
    
    try {
        auto startTime = std::chrono::high_resolution_clock::now();
        
        // Create inference engine if not exists
        if (!m_inferenceHandle) {
            // Try to create inference engine from compressed model
            QString compressedPath = m_currentModelPath;
            if (!compressedPath.endsWith(".sdr")) {
                compressedPath = m_currentModelPath + ".sdr";
                
                // Compress the model first
                CortexCompressionOptions options;
                cortex_compression_options_init(&options);
                options.sparsity = 0.02f;
                options.compression_level = 6;
                
                CortexError error = cortex_compressor_create(
                    m_currentModelPath.toUtf8().constData(),
                    m_currentModelFormat.toUtf8().constData(),
                    &options,
                    &m_compressorHandle
                );
                
                if (error.code != 0) {
                    result.errorMessage = QString("Failed to create compressor: %1").arg(error.message);
                    return result;
                }
                
                error = cortex_compressor_compress(m_compressorHandle, compressedPath.toUtf8().constData());
                if (error.code != 0) {
                    result.errorMessage = QString("Failed to compress model: %1").arg(error.message);
                    return result;
                }
            }
            
            // Create inference engine
            CortexError error = cortex_inference_engine_create(
                compressedPath.toUtf8().constData(),
                &m_inferenceHandle
            );
            
            if (error.code != 0) {
                result.errorMessage = QString("Failed to create inference engine: %1").arg(error.message);
                return result;
            }
        }
        
        // Load and preprocess audio
        std::vector<float> audioData = loadAudioFile(audioPath);
        std::vector<float> inputTensor = preprocessAudio(audioData);
        
        // Prepare output buffer
        std::vector<float> outputTensor(inputTensor.size());
        
        size_t actualOutputSize;
        CortexError error = cortex_inference_engine_run(
            m_inferenceHandle,
            inputTensor.data(),
            inputTensor.size(),
            outputTensor.data(),
            outputTensor.size(),
            &actualOutputSize
        );
        
        if (error.code != 0) {
            result.errorMessage = QString("Audio inference failed: %1").arg(error.message);
            return result;
        }
        
        // Resize output tensor to actual size
        outputTensor.resize(actualOutputSize);
        
        // Process audio output
        result.outputText = processAudioOutput(outputTensor, audioPath);
        
        auto endTime = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        result.inferenceTimeMs = duration.count();
        
        // Get memory usage
        result.memoryUsageMB = getCurrentMemoryUsage();
        result.success = true;
        
    } catch (const std::exception &e) {
        result.errorMessage = QString("Exception: %1").arg(e.what());
    }
    
    return result;
}

BenchmarkResult ModelManager::performBenchmark(int numRuns)
{
    BenchmarkResult result;
    result.modelPath = m_currentModelPath;
    result.numRuns = numRuns;
    
    try {
        // Run compression benchmarks
        for (int i = 0; i < numRuns; ++i) {
            QString tempOutput = QString("temp_benchmark_%1.sdr").arg(i);
            
            auto startTime = std::chrono::high_resolution_clock::now();
            CompressionResult compResult = performCompression(tempOutput, 0.02f, 6, false, 8);
            auto endTime = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
            
            if (compResult.success) {
                result.compressionTimes.append(compResult.compressionTimeMs);
                result.compressionRatios.append(compResult.compressionRatio);
            }
            
            // Clean up temp file
            QFile::remove(tempOutput);
        }
        
        // Run inference benchmarks
        if (m_inferenceHandle) {
            std::vector<float> testInput(128, 0.1f);
            std::vector<double> inferenceTimes;
            
            for (int i = 0; i < numRuns; ++i) {
                std::vector<float> outputTensor(testInput.size());
                size_t actualOutputSize;
                
                auto startTime = std::chrono::high_resolution_clock::now();
                CortexError error = cortex_inference_engine_run(
                    m_inferenceHandle,
                    testInput.data(),
                    testInput.size(),
                    outputTensor.data(),
                    outputTensor.size(),
                    &actualOutputSize
                );
                auto endTime = std::chrono::high_resolution_clock::now();
                
                if (error.code == 0) {
                    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
                    inferenceTimes.push_back(duration.count());
                }
            }
            
            result.inferenceTimes = QList<double>(inferenceTimes.begin(), inferenceTimes.end());
        }
        
        result.averageMemoryUsage = getCurrentMemoryUsage();
        
        // Calculate averages
        if (!result.compressionTimes.isEmpty()) {
            result.averageCompressionTime = std::accumulate(result.compressionTimes.begin(), 
                                                          result.compressionTimes.end(), 0.0) / result.compressionTimes.size();
        }
        if (!result.inferenceTimes.isEmpty()) {
            result.averageInferenceTime = std::accumulate(result.inferenceTimes.begin(), 
                                                        result.inferenceTimes.end(), 0.0) / result.inferenceTimes.size();
        }
        if (!result.memoryUsage.isEmpty()) {
            result.averageMemoryUsage = std::accumulate(result.memoryUsage.begin(), 
                                                      result.memoryUsage.end(), 0.0) / result.memoryUsage.size();
        }
        if (!result.compressionRatios.isEmpty()) {
            result.averageCompressionRatio = std::accumulate(result.compressionRatios.begin(), 
                                                           result.compressionRatios.end(), 0.0) / result.compressionRatios.size();
        }
    } catch (const std::exception &e) {
        // Handle error
        result.modelPath = m_currentModelPath;
        result.numRuns = 0;
    }
    
    return result;
}

void ModelManager::onCompressionFinished()
{
    CompressionResult result = m_compressionWatcher.result();
    emit compressionCompleted(result);
}

void ModelManager::onInferenceFinished()
{
    InferenceResult result = m_inferenceWatcher.result();
    emit inferenceCompleted(result);
}

void ModelManager::onBenchmarkFinished()
{
    BenchmarkResult result = m_benchmarkWatcher.result();
    emit benchmarkCompleted(result);
}

void ModelManager::handleError(const CortexError &error)
{
    emit this->error(getErrorMessage(error));
}

QString ModelManager::getErrorMessage(const CortexError &error)
{
    if (error.message) {
        return QString::fromUtf8(error.message);
    }
    return QString("Unknown error (code: %1)").arg(error.code);
} 

// Helper methods for inference
std::vector<float> ModelManager::textToTensor(const QString &text)
{
    std::vector<float> tensor;
    tensor.reserve(text.length() * 128); // Estimate size
    
    // Simple character-based encoding
    for (QChar ch : text) {
        // Convert character to ASCII and normalize
        float value = static_cast<float>(ch.unicode()) / 255.0f;
        tensor.push_back(value);
        
        // Add some context (simple sliding window)
        if (tensor.size() < 128) {
            tensor.push_back(0.0f);
        }
    }
    
    // Pad to minimum size
    while (tensor.size() < 128) {
        tensor.push_back(0.0f);
    }
    
    return tensor;
}

QString ModelManager::tensorToText(const std::vector<float> &tensor, int maxLength)
{
    QString result;
    result.reserve(maxLength);
    
    // Simple decoding: find peaks and convert back to characters
    for (size_t i = 0; i < tensor.size() && result.length() < maxLength; i += 2) {
        if (i + 1 < tensor.size()) {
            float value = tensor[i] * 255.0f;
            int charCode = static_cast<int>(value + 0.5f);
            
            if (charCode >= 32 && charCode <= 126) { // Printable ASCII
                result += QChar(charCode);
            } else if (charCode == 10) { // Newline
                result += '\n';
            } else if (charCode == 32) { // Space
                result += ' ';
            }
        }
    }
    
    if (result.isEmpty()) {
        result = "Generated text output";
    }
    
    return result;
}

std::vector<float> ModelManager::loadAudioFile(const QString &audioPath)
{
    std::vector<float> audioData;
    
    // Simple WAV file loader (basic implementation)
    QFile file(audioPath);
    if (!file.open(QIODevice::ReadOnly)) {
        throw std::runtime_error("Cannot open audio file");
    }
    
    // Read WAV header (simplified)
    QDataStream stream(&file);
    stream.setByteOrder(QDataStream::LittleEndian);
    
    // Skip WAV header (44 bytes)
    file.seek(44);
    
    // Read audio data as 16-bit samples
    while (!file.atEnd()) {
        qint16 sample;
        stream >> sample;
        audioData.push_back(static_cast<float>(sample) / 32768.0f); // Normalize to [-1, 1]
    }
    
    return audioData;
}

std::vector<float> ModelManager::preprocessAudio(const std::vector<float> &audioData)
{
    std::vector<float> processed;
    processed.reserve(audioData.size());
    
    // Simple preprocessing: normalize and apply basic filtering
    float maxAmplitude = 0.0f;
    for (float sample : audioData) {
        maxAmplitude = std::max(maxAmplitude, std::abs(sample));
    }
    
    if (maxAmplitude > 0.0f) {
        for (float sample : audioData) {
            processed.push_back(sample / maxAmplitude);
        }
    } else {
        processed = audioData;
    }
    
    return processed;
}

QString ModelManager::processAudioOutput(const std::vector<float> &outputTensor, const QString &originalPath)
{
    QString result = "Audio Processing Results:\n\n";
    
    // Analyze output tensor
    float maxVal = 0.0f;
    float minVal = 0.0f;
    float avgVal = 0.0f;
    
    for (float val : outputTensor) {
        maxVal = std::max(maxVal, val);
        minVal = std::min(minVal, val);
        avgVal += val;
    }
    
    if (!outputTensor.empty()) {
        avgVal /= outputTensor.size();
    }
    
    result += QString("Output Statistics:\n");
    result += QString("  - Max value: %1\n").arg(maxVal);
    result += QString("  - Min value: %1\n").arg(minVal);
    result += QString("  - Average value: %1\n").arg(avgVal);
    result += QString("  - Output size: %1 samples\n").arg(outputTensor.size());
    result += QString("  - Original file: %1\n").arg(QFileInfo(originalPath).fileName());
    
    return result;
}

double ModelManager::getCurrentMemoryUsage()
{
    // Get current process memory usage
    QProcess process;
    process.start("ps", QStringList() << "-o" << "rss=" << QString::number(QCoreApplication::applicationPid()));
    process.waitForFinished();
    
    QString output = process.readAllStandardOutput().trimmed();
    bool ok;
    double memoryKB = output.toDouble(&ok);
    
    if (ok) {
        return memoryKB / 1024.0; // Convert KB to MB
    }
    
    return 0.0;
} 