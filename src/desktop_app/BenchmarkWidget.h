#ifndef BENCHMARKWIDGET_H
#define BENCHMARKWIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QComboBox>
#include <QTextEdit>
#include <QProgressBar>
#include <QTableWidget>
#include <QHeaderView>
#include <QCheckBox>
#include <QFileDialog>

#include "ModelManager.h"

class BenchmarkWidget : public QWidget
{
    Q_OBJECT

public:
    explicit BenchmarkWidget(ModelManager *modelManager, QWidget *parent = nullptr);

private slots:
    void onStartBenchmark();
    void onBenchmarkStarted();
    void onBenchmarkProgress(int percentage);
    void onBenchmarkCompleted(const BenchmarkResult &result);
    void onSaveResults();
    void onClearResults();
    void onNumRunsChanged(int value);
    void onModelLoaded(const QString &modelPath);

private:
    void setupUI();
    void setupCharts();
    void updateResults(const BenchmarkResult &result);
    void enableControls(bool enable);
    void addBenchmarkRow(const QString &metric, double value, const QString &unit);
    
    // Model Manager
    ModelManager *m_modelManager;
    
    // UI Components - Settings
    QGroupBox *m_settingsGroup;
    QSpinBox *m_numRunsSpinBox;
    QLabel *m_numRunsLabel;
    
    QComboBox *m_benchmarkTypeCombo;
    QLabel *m_benchmarkTypeLabel;
    
    QCheckBox *m_includeCompressionCheck;
    QCheckBox *m_includeInferenceCheck;
    QCheckBox *m_includeMemoryCheck;
    
    // UI Components - Progress
    QGroupBox *m_progressGroup;
    QProgressBar *m_progressBar;
    QLabel *m_statusLabel;
    QLabel *m_currentRunLabel;
    
    // UI Components - Results Table
    QGroupBox *m_resultsGroup;
    QTableWidget *m_resultsTable;
    
    // UI Components - Charts
    QGroupBox *m_chartsGroup;
    
    // UI Components - Actions
    QPushButton *m_startButton;
    QPushButton *m_saveButton;
    QPushButton *m_clearButton;
    
    // State
    bool m_benchmarking;
    BenchmarkResult m_lastResult;
    QList<BenchmarkResult> m_benchmarkResults;
    
    // Chart update functions
    void updateCompressionChart();
    void updateInferenceChart();
    void updateMemoryChart();
};

#endif // BENCHMARKWIDGET_H 