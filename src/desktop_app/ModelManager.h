#ifndef MODELMANAGER_H
#define MODELMANAGER_H

#include <QObject>
#include <QString>
#include <QThread>
#include <QMutex>
#include <QFuture>
#include <QFutureWatcher>
#include <QtConcurrent>

#include "ai_compression/api/c_api.hpp"
#include "ai_compression/api/cortex_sdk.h"

struct CompressionResult {
    QString modelPath;
    QString outputPath;
    size_t originalSize;
    size_t compressedSize;
    double compressionRatio;
    double compressionTimeMs;
    bool success;
    QString errorMessage;
};

struct InferenceResult {
    QString modelPath;
    QString inputText;
    QString outputText;
    double inferenceTimeMs;
    double memoryUsageMB;
    bool success;
    QString errorMessage;
};

struct BenchmarkResult {
    QString modelPath;
    QList<double> compressionTimes;
    QList<double> inferenceTimes;
    QList<double> memoryUsage;
    QList<double> compressionRatios;
    double averageCompressionTime;
    double averageInferenceTime;
    double averageMemoryUsage;
    double averageCompressionRatio;
    int numRuns;
};

class ModelManager : public QObject
{
    Q_OBJECT

public:
    explicit ModelManager(QObject *parent = nullptr);
    ~ModelManager();

    // Model loading
    bool loadModel(const QString &modelPath, const QString &format = "auto");
    bool loadModelFromURL(const QString &url);
    
    // Compression
    void compressModel(const QString &outputPath, 
                      float sparsity = 0.02f,
                      int compressionLevel = 6,
                      bool useQuantization = false,
                      int quantizationBits = 8);
    
    // Inference
    void runTextInference(const QString &inputText, int maxLength = 100);
    void runAudioInference(const QString &audioPath);
    
    // Benchmarking
    void runBenchmark(int numRuns = 5);
    
    // Getters
    QString getCurrentModelPath() const { return m_currentModelPath; }
    QString getCurrentModelFormat() const { return m_currentModelFormat; }
    bool isModelLoaded() const { return m_modelLoaded; }

signals:
    void modelLoaded(const QString &modelPath);
    void modelLoadFailed(const QString &error);
    void compressionStarted();
    void compressionProgress(int percentage);
    void compressionCompleted(const CompressionResult &result);
    void inferenceStarted();
    void inferenceCompleted(const InferenceResult &result);
    void benchmarkStarted();
    void benchmarkProgress(int percentage);
    void benchmarkCompleted(const BenchmarkResult &result);
    void error(const QString &error);

private slots:
    void onCompressionFinished();
    void onInferenceFinished();
    void onBenchmarkFinished();

private:
    // Model state
    QString m_currentModelPath;
    QString m_currentModelFormat;
    bool m_modelLoaded;
    
    // SDK handles
    CortexCompressorHandle m_compressorHandle;
    CortexInferenceEngineHandle m_inferenceHandle;
    
    // Threading
    QFutureWatcher<CompressionResult> m_compressionWatcher;
    QFutureWatcher<InferenceResult> m_inferenceWatcher;
    QFutureWatcher<BenchmarkResult> m_benchmarkWatcher;
    
    // Thread safety
    mutable QMutex m_mutex;
    
    // Helper methods
    QString detectModelFormat(const QString &modelPath);
    CompressionResult performCompression(const QString &outputPath, 
                                       float sparsity,
                                       int compressionLevel,
                                       bool useQuantization,
                                       int quantizationBits);
    InferenceResult performTextInference(const QString &inputText, int maxLength);
    InferenceResult performAudioInference(const QString &audioPath);
    BenchmarkResult performBenchmark(int numRuns);
    
    // Error handling
    void handleError(const CortexError &error);
    QString getErrorMessage(const CortexError &error);
    
    // Helper methods for inference
    std::vector<float> textToTensor(const QString &text);
    QString tensorToText(const std::vector<float> &tensor, int maxLength);
    std::vector<float> loadAudioFile(const QString &audioPath);
    std::vector<float> preprocessAudio(const std::vector<float> &audioData);
    QString processAudioOutput(const std::vector<float> &outputTensor, const QString &originalPath);
    double getCurrentMemoryUsage();
};

#endif // MODELMANAGER_H 