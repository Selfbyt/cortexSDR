#ifndef RESULTSWIDGET_H
#define RESULTSWIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QPushButton>
#include <QTextEdit>
#include <QTableWidget>
#include <QHeaderView>
#include <QFileDialog>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>

#include "ModelManager.h"

class ResultsWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ResultsWidget(QWidget *parent = nullptr);

public slots:
    void addCompressionResult(const CompressionResult &result);
    void addInferenceResult(const InferenceResult &result);
    void addBenchmarkResult(const BenchmarkResult &result);
    void clearResults();

private slots:
    void onSaveResults();
    void onExportCSV();
    void onExportJSON();
    void onClearResults();
    void onShowCompressionChart();
    void onShowInferenceChart();
    void onShowBenchmarkChart();

private:
    void setupUI();
    void setupCharts();
    void updateCompressionChart();
    void updateInferenceChart();
    void updateBenchmarkChart();
    void saveToFile(const QString &filename, const QString &content);
    void updateSummary();
    
    // UI Components - Summary
    QGroupBox *m_summaryGroup;
    QLabel *m_totalModelsLabel;
    QLabel *m_totalCompressionsLabel;
    QLabel *m_totalInferencesLabel;
    QLabel *m_totalBenchmarksLabel;
    QLabel *m_averageCompressionRatioLabel;
    QLabel *m_averageInferenceTimeLabel;
    
    // UI Components - Results Table
    QGroupBox *m_resultsGroup;
    QTableWidget *m_resultsTable;
    
    // UI Components - Charts
    QGroupBox *m_chartsGroup;
    QTabWidget *m_chartTabs;
    
    // UI Components - Actions
    QPushButton *m_saveButton;
    QPushButton *m_exportCSVButton;
    QPushButton *m_exportJSONButton;
    QPushButton *m_clearButton;
    
    // Data storage
    QList<CompressionResult> m_compressionResults;
    QList<InferenceResult> m_inferenceResults;
    QList<BenchmarkResult> m_benchmarkResults;
};

#endif // RESULTSWIDGET_H 