#include "BenchmarkWidget.h"
#include <QLineSeries>
#include <QMessageBox>
#include <QStandardPaths>

BenchmarkWidget::BenchmarkWidget(ModelManager *modelManager, QWidget *parent)
    : QWidget(parent)
    , m_modelManager(modelManager)
    , m_benchmarking(false)
{
    setupUI();
    
    if (m_modelManager) {
        connect(m_modelManager, &ModelManager::benchmarkStarted, this, &BenchmarkWidget::onBenchmarkStarted);
        connect(m_modelManager, &ModelManager::benchmarkProgress, this, &BenchmarkWidget::onBenchmarkProgress);
        connect(m_modelManager, &ModelManager::benchmarkCompleted, this, &BenchmarkWidget::onBenchmarkCompleted);
        connect(m_modelManager, &ModelManager::modelLoaded, this, &BenchmarkWidget::onModelLoaded);
    }
}

void BenchmarkWidget::setupUI()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    
    // Settings Group
    m_settingsGroup = new QGroupBox("Benchmark Settings", this);
    QGridLayout *settingsLayout = new QGridLayout(m_settingsGroup);
    
    settingsLayout->addWidget(new QLabel("Number of Runs:"), 0, 0);
    m_numRunsSpinBox = new QSpinBox(this);
    m_numRunsSpinBox->setRange(1, 100);
    m_numRunsSpinBox->setValue(5);
    connect(m_numRunsSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), 
            this, &BenchmarkWidget::onNumRunsChanged);
    settingsLayout->addWidget(m_numRunsSpinBox, 0, 1);
    
    settingsLayout->addWidget(new QLabel("Benchmark Type:"), 1, 0);
    m_benchmarkTypeCombo = new QComboBox(this);
    m_benchmarkTypeCombo->addItems({"Compression", "Inference", "Both"});
    settingsLayout->addWidget(m_benchmarkTypeCombo, 1, 1);
    
    m_includeCompressionCheck = new QCheckBox("Include Compression", this);
    m_includeCompressionCheck->setChecked(true);
    settingsLayout->addWidget(m_includeCompressionCheck, 2, 0);
    
    m_includeInferenceCheck = new QCheckBox("Include Inference", this);
    m_includeInferenceCheck->setChecked(true);
    settingsLayout->addWidget(m_includeInferenceCheck, 2, 1);
    
    m_includeMemoryCheck = new QCheckBox("Include Memory Usage", this);
    m_includeMemoryCheck->setChecked(true);
    settingsLayout->addWidget(m_includeMemoryCheck, 2, 2);
    
    mainLayout->addWidget(m_settingsGroup);
    
    // Progress Group
    m_progressGroup = new QGroupBox("Progress", this);
    QVBoxLayout *progressLayout = new QVBoxLayout(m_progressGroup);
    
    m_progressBar = new QProgressBar(this);
    m_progressBar->setVisible(false);
    progressLayout->addWidget(m_progressBar);
    
    m_statusLabel = new QLabel("Ready", this);
    progressLayout->addWidget(m_statusLabel);
    
    m_currentRunLabel = new QLabel("", this);
    progressLayout->addWidget(m_currentRunLabel);
    
    mainLayout->addWidget(m_progressGroup);
    
    // Results Group
    m_resultsGroup = new QGroupBox("Results", this);
    QVBoxLayout *resultsLayout = new QVBoxLayout(m_resultsGroup);
    
    m_resultsTable = new QTableWidget(this);
    m_resultsTable->setColumnCount(3);
    m_resultsTable->setHorizontalHeaderLabels({"Metric", "Value", "Unit"});
    m_resultsTable->horizontalHeader()->setStretchLastSection(true);
    resultsLayout->addWidget(m_resultsTable);
    
    mainLayout->addWidget(m_resultsGroup);
    
    // Charts Group
    m_chartsGroup = new QGroupBox("Charts", this);
    QVBoxLayout *chartsLayout = new QVBoxLayout(m_chartsGroup);
    
    // For now, just add placeholder labels
    chartsLayout->addWidget(new QLabel("Charts will be displayed here when Qt Charts is available"));
    
    mainLayout->addWidget(m_chartsGroup);
    
    // Actions
    QHBoxLayout *actionsLayout = new QHBoxLayout();
    
    m_startButton = new QPushButton("Start Benchmark", this);
    m_startButton->setEnabled(false);
    connect(m_startButton, &QPushButton::clicked, this, &BenchmarkWidget::onStartBenchmark);
    actionsLayout->addWidget(m_startButton);
    
    m_saveButton = new QPushButton("Save Results", this);
    m_saveButton->setEnabled(false);
    connect(m_saveButton, &QPushButton::clicked, this, &BenchmarkWidget::onSaveResults);
    actionsLayout->addWidget(m_saveButton);
    
    m_clearButton = new QPushButton("Clear Results", this);
    connect(m_clearButton, &QPushButton::clicked, this, &BenchmarkWidget::onClearResults);
    actionsLayout->addWidget(m_clearButton);
    
    actionsLayout->addStretch();
    mainLayout->addLayout(actionsLayout);
}

void BenchmarkWidget::onStartBenchmark()
{
    if (!m_modelManager || m_benchmarking) return;
    
    int numRuns = m_numRunsSpinBox->value();
    m_modelManager->runBenchmark(numRuns);
}

void BenchmarkWidget::onBenchmarkStarted()
{
    m_benchmarking = true;
    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, 100);
    m_progressBar->setValue(0);
    m_statusLabel->setText("Benchmark started...");
    m_currentRunLabel->setText("Run 1 of " + QString::number(m_numRunsSpinBox->value()));
    enableControls(false);
}

void BenchmarkWidget::onBenchmarkProgress(int percentage)
{
    m_progressBar->setValue(percentage);
    m_statusLabel->setText(QString("Benchmarking... %1%").arg(percentage));
    
    int currentRun = (percentage * m_numRunsSpinBox->value()) / 100 + 1;
    m_currentRunLabel->setText(QString("Run %1 of %2").arg(currentRun).arg(m_numRunsSpinBox->value()));
}

void BenchmarkWidget::onBenchmarkCompleted(const BenchmarkResult &result)
{
    m_benchmarking = false;
    m_progressBar->setVisible(false);
    m_statusLabel->setText("Benchmark completed");
    m_currentRunLabel->setText("");
    enableControls(true);
    
    m_lastResult = result;
    m_benchmarkResults.append(result);
    
    updateResults(result);
    
    // Update charts
    updateCompressionChart();
    updateInferenceChart();
    updateMemoryChart();
    
    m_saveButton->setEnabled(true);
}

void BenchmarkWidget::onSaveResults()
{
    QString filename = QFileDialog::getSaveFileName(
        this,
        "Save Benchmark Results",
        QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation),
        "CSV Files (*.csv);;JSON Files (*.json);;All Files (*.*)"
    );
    
    if (!filename.isEmpty()) {
        // TODO: Implement saving results
    }
}

void BenchmarkWidget::onClearResults()
{
    m_resultsTable->setRowCount(0);
    m_benchmarkResults.clear();
    m_saveButton->setEnabled(false);
    m_lastResult = BenchmarkResult();
    
    // Clear charts
    updateCompressionChart();
    updateInferenceChart();
    updateMemoryChart();
}

void BenchmarkWidget::onNumRunsChanged(int value)
{
    // Update UI if needed
}

void BenchmarkWidget::onModelLoaded(const QString &modelPath)
{
    m_startButton->setEnabled(true);
}



void BenchmarkWidget::setupCharts()
{
    // Create placeholder labels for charts
    QVBoxLayout *chartsLayout = new QVBoxLayout(m_chartsGroup);
    
    QLabel *compressionLabel = new QLabel("Compression Chart (Qt Charts not available)", this);
    compressionLabel->setAlignment(Qt::AlignCenter);
    compressionLabel->setMinimumHeight(100);
    compressionLabel->setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ccc; }");
    chartsLayout->addWidget(compressionLabel);
    
    QLabel *inferenceLabel = new QLabel("Inference Chart (Qt Charts not available)", this);
    inferenceLabel->setAlignment(Qt::AlignCenter);
    inferenceLabel->setMinimumHeight(100);
    inferenceLabel->setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ccc; }");
    chartsLayout->addWidget(inferenceLabel);
    
    QLabel *memoryLabel = new QLabel("Memory Chart (Qt Charts not available)", this);
    memoryLabel->setAlignment(Qt::AlignCenter);
    memoryLabel->setMinimumHeight(100);
    memoryLabel->setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ccc; }");
    chartsLayout->addWidget(memoryLabel);
} 

void BenchmarkWidget::updateResults(const BenchmarkResult &result)
{
    m_resultsTable->setRowCount(0);
    
    addBenchmarkRow("Number of Runs", result.numRuns, "");
    addBenchmarkRow("Average Compression Time", result.averageCompressionTime, "ms");
    addBenchmarkRow("Average Compression Ratio", result.averageCompressionRatio, "x");
    addBenchmarkRow("Average Inference Time", result.averageInferenceTime, "ms");
    addBenchmarkRow("Average Memory Usage", result.averageMemoryUsage, "MB");
}

void BenchmarkWidget::enableControls(bool enable)
{
    m_startButton->setEnabled(enable);
    m_saveButton->setEnabled(enable && !m_lastResult.modelPath.isEmpty());
    m_clearButton->setEnabled(enable);
    m_numRunsSpinBox->setEnabled(enable);
    m_benchmarkTypeCombo->setEnabled(enable);
    m_includeCompressionCheck->setEnabled(enable);
    m_includeInferenceCheck->setEnabled(enable);
    m_includeMemoryCheck->setEnabled(enable);
}

void BenchmarkWidget::addBenchmarkRow(const QString &metric, double value, const QString &unit)
{
    int row = m_resultsTable->rowCount();
    m_resultsTable->insertRow(row);
    
    m_resultsTable->setItem(row, 0, new QTableWidgetItem(metric));
    m_resultsTable->setItem(row, 1, new QTableWidgetItem(QString::number(value, 'f', 2)));
    m_resultsTable->setItem(row, 2, new QTableWidgetItem(unit));
} 

void BenchmarkWidget::updateCompressionChart()
{
    // TODO: Implement when Qt Charts is available
}

void BenchmarkWidget::updateInferenceChart()
{
    // TODO: Implement when Qt Charts is available
}

void BenchmarkWidget::updateMemoryChart()
{
    // TODO: Implement when Qt Charts is available
} 