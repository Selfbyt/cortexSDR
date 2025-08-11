#include "ResultsWidget.h"
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QPushButton>
#include <QTableWidget>
#include <QHeaderView>
#include <QFileDialog>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QMessageBox>
#include <QDateTime>
#include <QFileInfo>
#include <QStandardPaths>

ResultsWidget::ResultsWidget(QWidget *parent)
    : QWidget(parent)
{
    setupUI();
}

void ResultsWidget::setupUI()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    
    // Summary Group
    m_summaryGroup = new QGroupBox("Summary", this);
    QGridLayout *summaryLayout = new QGridLayout(m_summaryGroup);
    
    summaryLayout->addWidget(new QLabel("Total Models:"), 0, 0);
    m_totalModelsLabel = new QLabel("0", this);
    summaryLayout->addWidget(m_totalModelsLabel, 0, 1);
    
    summaryLayout->addWidget(new QLabel("Total Compressions:"), 0, 2);
    m_totalCompressionsLabel = new QLabel("0", this);
    summaryLayout->addWidget(m_totalCompressionsLabel, 0, 3);
    
    summaryLayout->addWidget(new QLabel("Total Inferences:"), 1, 0);
    m_totalInferencesLabel = new QLabel("0", this);
    summaryLayout->addWidget(m_totalInferencesLabel, 1, 1);
    
    summaryLayout->addWidget(new QLabel("Total Benchmarks:"), 1, 2);
    m_totalBenchmarksLabel = new QLabel("0", this);
    summaryLayout->addWidget(m_totalBenchmarksLabel, 1, 3);
    
    summaryLayout->addWidget(new QLabel("Avg Compression Ratio:"), 2, 0);
    m_averageCompressionRatioLabel = new QLabel("0.00x", this);
    summaryLayout->addWidget(m_averageCompressionRatioLabel, 2, 1);
    
    summaryLayout->addWidget(new QLabel("Avg Inference Time:"), 2, 2);
    m_averageInferenceTimeLabel = new QLabel("0.00 ms", this);
    summaryLayout->addWidget(m_averageInferenceTimeLabel, 2, 3);
    
    mainLayout->addWidget(m_summaryGroup);
    
    // Results Table
    m_resultsGroup = new QGroupBox("All Results", this);
    QVBoxLayout *resultsLayout = new QVBoxLayout(m_resultsGroup);
    
    m_resultsTable = new QTableWidget(this);
    m_resultsTable->setColumnCount(6);
    m_resultsTable->setHorizontalHeaderLabels({"Type", "Model", "Time", "Size", "Ratio", "Status"});
    m_resultsTable->horizontalHeader()->setStretchLastSection(true);
    resultsLayout->addWidget(m_resultsTable);
    
    mainLayout->addWidget(m_resultsGroup);
    
    // Charts Group
    m_chartsGroup = new QGroupBox("Charts", this);
    QVBoxLayout *chartsLayout = new QVBoxLayout(m_chartsGroup);
    
    m_chartTabs = new QTabWidget(this);
    
    // Placeholder for charts
    QWidget *compressionChartPlaceholder = new QWidget(this);
    QVBoxLayout *compressionChartLayout = new QVBoxLayout(compressionChartPlaceholder);
    compressionChartLayout->addWidget(new QLabel("Compression Performance Chart"));
    m_chartTabs->addTab(compressionChartPlaceholder, "Compression");
    
    QWidget *inferenceChartPlaceholder = new QWidget(this);
    QVBoxLayout *inferenceChartLayout = new QVBoxLayout(inferenceChartPlaceholder);
    inferenceChartLayout->addWidget(new QLabel("Inference Performance Chart"));
    m_chartTabs->addTab(inferenceChartPlaceholder, "Inference");
    
    QWidget *benchmarkChartPlaceholder = new QWidget(this);
    QVBoxLayout *benchmarkChartLayout = new QVBoxLayout(benchmarkChartPlaceholder);
    benchmarkChartLayout->addWidget(new QLabel("Benchmark Results Chart"));
    m_chartTabs->addTab(benchmarkChartPlaceholder, "Benchmark");
    
    chartsLayout->addWidget(m_chartTabs);
    
    mainLayout->addWidget(m_chartsGroup);
    
    // Actions
    QHBoxLayout *actionsLayout = new QHBoxLayout();
    
    m_saveButton = new QPushButton("Save Results", this);
    connect(m_saveButton, &QPushButton::clicked, this, &ResultsWidget::onSaveResults);
    actionsLayout->addWidget(m_saveButton);
    
    m_exportCSVButton = new QPushButton("Export CSV", this);
    connect(m_exportCSVButton, &QPushButton::clicked, this, &ResultsWidget::onExportCSV);
    actionsLayout->addWidget(m_exportCSVButton);
    
    m_exportJSONButton = new QPushButton("Export JSON", this);
    connect(m_exportJSONButton, &QPushButton::clicked, this, &ResultsWidget::onExportJSON);
    actionsLayout->addWidget(m_exportJSONButton);
    
    m_clearButton = new QPushButton("Clear All", this);
    connect(m_clearButton, &QPushButton::clicked, this, &ResultsWidget::onClearResults);
    actionsLayout->addWidget(m_clearButton);
    
    actionsLayout->addStretch();
    mainLayout->addLayout(actionsLayout);
}

void ResultsWidget::addCompressionResult(const CompressionResult &result)
{
    m_compressionResults.append(result);
    updateSummary();
    
    // Add to table
    int row = m_resultsTable->rowCount();
    m_resultsTable->insertRow(row);
    
    m_resultsTable->setItem(row, 0, new QTableWidgetItem("Compression"));
    m_resultsTable->setItem(row, 1, new QTableWidgetItem(QFileInfo(result.modelPath).fileName()));
    m_resultsTable->setItem(row, 2, new QTableWidgetItem(QString::number(result.compressionTimeMs, 'f', 1) + " ms"));
    m_resultsTable->setItem(row, 3, new QTableWidgetItem(QString::number(result.compressedSize)));
    m_resultsTable->setItem(row, 4, new QTableWidgetItem(QString::number(result.compressionRatio, 'f', 2) + "x"));
    m_resultsTable->setItem(row, 5, new QTableWidgetItem(result.success ? "Success" : "Failed"));
    
    // Update chart
    updateCompressionChart();
}

void ResultsWidget::addInferenceResult(const InferenceResult &result)
{
    m_inferenceResults.append(result);
    updateSummary();
    
    // Add to table
    int row = m_resultsTable->rowCount();
    m_resultsTable->insertRow(row);
    
    m_resultsTable->setItem(row, 0, new QTableWidgetItem("Inference"));
    m_resultsTable->setItem(row, 1, new QTableWidgetItem(QFileInfo(result.modelPath).fileName()));
    m_resultsTable->setItem(row, 2, new QTableWidgetItem(QString::number(result.inferenceTimeMs, 'f', 1) + " ms"));
    m_resultsTable->setItem(row, 3, new QTableWidgetItem(QString::number(result.memoryUsageMB, 'f', 1) + " MB"));
    m_resultsTable->setItem(row, 4, new QTableWidgetItem("--"));
    m_resultsTable->setItem(row, 5, new QTableWidgetItem(result.success ? "Success" : "Failed"));
    
    // Update chart
    updateInferenceChart();
}

void ResultsWidget::addBenchmarkResult(const BenchmarkResult &result)
{
    m_benchmarkResults.append(result);
    updateSummary();
    
    // Add to table
    int row = m_resultsTable->rowCount();
    m_resultsTable->insertRow(row);
    
    m_resultsTable->setItem(row, 0, new QTableWidgetItem("Benchmark"));
    m_resultsTable->setItem(row, 1, new QTableWidgetItem(QFileInfo(result.modelPath).fileName()));
    m_resultsTable->setItem(row, 2, new QTableWidgetItem(QString::number(result.averageCompressionTime, 'f', 1) + " ms"));
    m_resultsTable->setItem(row, 3, new QTableWidgetItem(QString::number(result.averageMemoryUsage, 'f', 1) + " MB"));
    m_resultsTable->setItem(row, 4, new QTableWidgetItem(QString::number(result.averageCompressionRatio, 'f', 2) + "x"));
    m_resultsTable->setItem(row, 5, new QTableWidgetItem("Completed"));
    
    // Update chart
    updateBenchmarkChart();
}

void ResultsWidget::clearResults()
{
    m_compressionResults.clear();
    m_inferenceResults.clear();
    m_benchmarkResults.clear();
    m_resultsTable->setRowCount(0);
    updateSummary();
}

void ResultsWidget::onSaveResults()
{
    QString filename = QFileDialog::getSaveFileName(
        this,
        "Save Results",
        QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation) + "/cortexsdr_results.json",
        "JSON Files (*.json);;Text Files (*.txt);;All Files (*.*)"
    );
    
    if (!filename.isEmpty()) {
        QJsonObject resultsData;
        
        // Add compression results
        QJsonArray compressionArray;
        for (const auto& result : m_compressionResults) {
            QJsonObject compObj;
            compObj["modelPath"] = result.modelPath;
            compObj["outputPath"] = result.outputPath;
            compObj["originalSize"] = static_cast<qint64>(result.originalSize);
            compObj["compressedSize"] = static_cast<qint64>(result.compressedSize);
            compObj["compressionRatio"] = result.compressionRatio;
            compObj["compressionTimeMs"] = result.compressionTimeMs;
            compObj["success"] = result.success;
            if (!result.success) {
                compObj["errorMessage"] = result.errorMessage;
            }
            compressionArray.append(compObj);
        }
        resultsData["compressionResults"] = compressionArray;
        
        // Add inference results
        QJsonArray inferenceArray;
        for (const auto& result : m_inferenceResults) {
            QJsonObject infObj;
            infObj["modelPath"] = result.modelPath;
            infObj["inputText"] = result.inputText;
            infObj["outputText"] = result.outputText;
            infObj["inferenceTimeMs"] = result.inferenceTimeMs;
            infObj["memoryUsageMB"] = result.memoryUsageMB;
            infObj["success"] = result.success;
            if (!result.success) {
                infObj["errorMessage"] = result.errorMessage;
            }
            inferenceArray.append(infObj);
        }
        resultsData["inferenceResults"] = inferenceArray;
        
        // Add benchmark results
        QJsonArray benchmarkArray;
        for (const auto& result : m_benchmarkResults) {
            QJsonObject benchObj;
            benchObj["modelPath"] = result.modelPath;
            benchObj["numRuns"] = result.numRuns;
            benchObj["averageCompressionTime"] = result.averageCompressionTime;
            benchObj["averageCompressionRatio"] = result.averageCompressionRatio;
            benchObj["averageInferenceTime"] = result.averageInferenceTime;
            benchObj["averageMemoryUsage"] = result.averageMemoryUsage;
            benchmarkArray.append(benchObj);
        }
        resultsData["benchmarkResults"] = benchmarkArray;
        
        // Add metadata
        resultsData["exportDate"] = QDateTime::currentDateTime().toString(Qt::ISODate);
        resultsData["totalResults"] = m_compressionResults.size() + m_inferenceResults.size() + m_benchmarkResults.size();
        
        // Save to file
        QJsonDocument doc(resultsData);
        QFile file(filename);
        if (file.open(QIODevice::WriteOnly)) {
            file.write(doc.toJson());
            QMessageBox::information(this, "Success", "Results saved successfully!");
        } else {
            QMessageBox::critical(this, "Error", "Failed to save results file.");
        }
    }
}

void ResultsWidget::onExportCSV()
{
    QString filename = QFileDialog::getSaveFileName(
        this,
        "Export CSV",
        QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation),
        "CSV Files (*.csv)"
    );
    
    if (!filename.isEmpty()) {
        // TODO: Implement CSV export
    }
}

void ResultsWidget::onExportJSON()
{
    QString filename = QFileDialog::getSaveFileName(
        this,
        "Export JSON",
        QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation),
        "JSON Files (*.json)"
    );
    
    if (!filename.isEmpty()) {
        // TODO: Implement JSON export
    }
}

void ResultsWidget::onClearResults()
{
    clearResults();
}

void ResultsWidget::onShowCompressionChart()
{
    // TODO: Implement when Qt Charts is available
}

void ResultsWidget::onShowInferenceChart()
{
    // TODO: Implement when Qt Charts is available
}

void ResultsWidget::onShowBenchmarkChart()
{
    // TODO: Implement when Qt Charts is available
}



void ResultsWidget::setupCharts()
{
    // Create placeholder labels for charts
    QLabel *compressionLabel = new QLabel("Compression Chart (Qt Charts not available)", this);
    compressionLabel->setAlignment(Qt::AlignCenter);
    compressionLabel->setMinimumHeight(200);
    compressionLabel->setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ccc; }");
    m_chartTabs->addTab(compressionLabel, "Compression");
    
    QLabel *inferenceLabel = new QLabel("Inference Chart (Qt Charts not available)", this);
    inferenceLabel->setAlignment(Qt::AlignCenter);
    inferenceLabel->setMinimumHeight(200);
    inferenceLabel->setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ccc; }");
    m_chartTabs->addTab(inferenceLabel, "Inference");
    
    QLabel *benchmarkLabel = new QLabel("Benchmark Chart (Qt Charts not available)", this);
    benchmarkLabel->setAlignment(Qt::AlignCenter);
    benchmarkLabel->setMinimumHeight(200);
    benchmarkLabel->setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ccc; }");
    m_chartTabs->addTab(benchmarkLabel, "Benchmark");
}

void ResultsWidget::updateCompressionChart()
{
    // TODO: Implement when Qt Charts is available
}

void ResultsWidget::updateInferenceChart()
{
    // TODO: Implement when Qt Charts is available
}

void ResultsWidget::updateBenchmarkChart()
{
    // TODO: Implement when Qt Charts is available
}

void ResultsWidget::saveToFile(const QString &filename, const QString &content)
{
    QFile file(filename);
    if (file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        QTextStream out(&file);
        out << content;
        file.close();
    }
}

void ResultsWidget::updateSummary()
{
    int totalModels = 0;
    int totalCompressions = m_compressionResults.size();
    int totalInferences = m_inferenceResults.size();
    int totalBenchmarks = m_benchmarkResults.size();
    
    // Count unique models
    QSet<QString> models;
    for (const auto &result : m_compressionResults) {
        models.insert(result.modelPath);
    }
    for (const auto &result : m_inferenceResults) {
        models.insert(result.modelPath);
    }
    for (const auto &result : m_benchmarkResults) {
        models.insert(result.modelPath);
    }
    totalModels = models.size();
    
    // Calculate averages
    double avgCompressionRatio = 0.0;
    double avgInferenceTime = 0.0;
    
    if (!m_compressionResults.isEmpty()) {
        double sum = 0.0;
        for (const auto &result : m_compressionResults) {
            if (result.success) {
                sum += result.compressionRatio;
            }
        }
        avgCompressionRatio = sum / m_compressionResults.size();
    }
    
    if (!m_inferenceResults.isEmpty()) {
        double sum = 0.0;
        for (const auto &result : m_inferenceResults) {
            if (result.success) {
                sum += result.inferenceTimeMs;
            }
        }
        avgInferenceTime = sum / m_inferenceResults.size();
    }
    
    // Update labels
    m_totalModelsLabel->setText(QString::number(totalModels));
    m_totalCompressionsLabel->setText(QString::number(totalCompressions));
    m_totalInferencesLabel->setText(QString::number(totalInferences));
    m_totalBenchmarksLabel->setText(QString::number(totalBenchmarks));
    m_averageCompressionRatioLabel->setText(QString::number(avgCompressionRatio, 'f', 2) + "x");
    m_averageInferenceTimeLabel->setText(QString::number(avgInferenceTime, 'f', 1) + " ms");
} 