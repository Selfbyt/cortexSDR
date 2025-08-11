#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTabWidget>
#include <QStatusBar>
#include <QProgressBar>
#include <QLabel>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QFileDialog>
#include <QMessageBox>
#include <QSettings>

#include "ModelManager.h"
#include "CompressionWidget.h"
#include "InferenceWidget.h"
#include "BenchmarkWidget.h"
#include "ResultsWidget.h"
#include "ChatWidget.h"
#include "PerformanceMonitor.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void onModelLoaded(const QString &modelPath);
    void onCompressionCompleted(const CompressionResult &result);
    void onInferenceCompleted(const InferenceResult &result);
    void onBenchmarkCompleted(const BenchmarkResult &result);
    void onPerformanceUpdate(const PerformanceMetrics &metrics);
    
    void openModel();
    void saveResults();
    void showAbout();
    void showSettings();
    void showModelManager();

private:
    void setupUI();
    void setupMenuBar();
    void setupStatusBar();
    void setupToolBar();
    void loadSettings();
    void saveSettings();
    void updateModelInfo();
    
    // UI Components
    QTabWidget *m_tabWidget;
    CompressionWidget *m_compressionWidget;
    InferenceWidget *m_inferenceWidget;
    BenchmarkWidget *m_benchmarkWidget;
    ResultsWidget *m_resultsWidget;
    ChatWidget *m_chatWidget;
    
    // Core components
    ModelManager *m_modelManager;
    PerformanceMonitor *m_performanceMonitor;
    
    // Status bar components
    QProgressBar *m_progressBar;
    QLabel *m_statusLabel;
    QLabel *m_memoryLabel;
    QLabel *m_cpuLabel;
    QLabel *m_modelLabel;
    
    // Toolbar components
    QAction *m_openModelAction;
    QAction *m_saveResultsAction;
    QAction *m_benchmarkAction;
    QAction *m_chatAction;
    
    // Settings
    QSettings m_settings;
};

#endif // MAINWINDOW_H 