#include "MainWindow.h"
#include <QApplication>
#include <QMessageBox>
#include <QStandardPaths>
#include <QDir>
#include <QVBoxLayout>
#include <QLabel>
#include <QToolBar>
#include <QFileInfo>
#include <QtConcurrent>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , m_tabWidget(nullptr)
    , m_compressionWidget(nullptr)
    , m_inferenceWidget(nullptr)
    , m_benchmarkWidget(nullptr)
    , m_resultsWidget(nullptr)
    , m_chatWidget(nullptr)
    , m_modelManager(nullptr)
    , m_performanceMonitor(nullptr)
    , m_progressBar(nullptr)
    , m_statusLabel(nullptr)
    , m_memoryLabel(nullptr)
    , m_cpuLabel(nullptr)
    , m_modelLabel(nullptr)
    , m_settings("CortexSDR", "DesktopApp")
{
    setupUI();
    setupMenuBar();
    setupToolBar();
    setupStatusBar();
    loadSettings();
    
    // Initialize core components
    m_modelManager = new ModelManager(this);
    m_performanceMonitor = new PerformanceMonitor(this);
    
    // Connect signals
    connect(m_modelManager, &ModelManager::modelLoaded, this, &MainWindow::onModelLoaded);
    connect(m_modelManager, &ModelManager::compressionCompleted, this, &MainWindow::onCompressionCompleted);
    connect(m_modelManager, &ModelManager::inferenceCompleted, this, &MainWindow::onInferenceCompleted);
    connect(m_modelManager, &ModelManager::benchmarkCompleted, this, &MainWindow::onBenchmarkCompleted);
    connect(m_performanceMonitor, &PerformanceMonitor::metricsUpdated, this, &MainWindow::onPerformanceUpdate);
    
    // Start performance monitoring
    m_performanceMonitor->startMonitoring(2000); // Update every 2 seconds
    
    // Set window properties
    setWindowTitle("CortexSDR Desktop - AI Model Compression & Inference");
    setMinimumSize(1400, 900);
    resize(1600, 1000);
}

MainWindow::~MainWindow()
{
    saveSettings();
    if (m_performanceMonitor) {
        m_performanceMonitor->stopMonitoring();
    }
}

void MainWindow::setupUI()
{
    // Create central widget
    QWidget *centralWidget = new QWidget(this);
    setCentralWidget(centralWidget);
    
    // Create main layout
    QVBoxLayout *mainLayout = new QVBoxLayout(centralWidget);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    
    // Create tab widget
    m_tabWidget = new QTabWidget(this);
    mainLayout->addWidget(m_tabWidget);
    
    // Create widgets
    m_compressionWidget = new CompressionWidget(m_modelManager, this);
    m_inferenceWidget = new InferenceWidget(m_modelManager, this);
    m_benchmarkWidget = new BenchmarkWidget(m_modelManager, this);
    m_resultsWidget = new ResultsWidget(this);
    m_chatWidget = new ChatWidget(m_modelManager, this);
    
    // Add tabs
    m_tabWidget->addTab(m_chatWidget, "💬 Chat");
    m_tabWidget->addTab(m_compressionWidget, "🗜️ Compression");
    m_tabWidget->addTab(m_inferenceWidget, "⚡ Inference");
    m_tabWidget->addTab(m_benchmarkWidget, "📊 Benchmark");
    m_tabWidget->addTab(m_resultsWidget, "📋 Results");
    
    // Set chat as default tab
    m_tabWidget->setCurrentIndex(0);
}

void MainWindow::setupMenuBar()
{
    // File menu
    QMenu *fileMenu = menuBar()->addMenu("&File");
    
    m_openModelAction = fileMenu->addAction("&Open Model...");
    m_openModelAction->setShortcut(QKeySequence::Open);
    connect(m_openModelAction, &QAction::triggered, this, &MainWindow::openModel);
    
    m_saveResultsAction = fileMenu->addAction("&Save Results...");
    m_saveResultsAction->setShortcut(QKeySequence::Save);
    connect(m_saveResultsAction, &QAction::triggered, this, &MainWindow::saveResults);
    
    fileMenu->addSeparator();
    
    QAction *exitAction = fileMenu->addAction("E&xit");
    exitAction->setShortcut(QKeySequence::Quit);
    connect(exitAction, &QAction::triggered, this, &QWidget::close);
    
    // Tools menu
    QMenu *toolsMenu = menuBar()->addMenu("&Tools");
    
    m_benchmarkAction = toolsMenu->addAction("&Run Benchmark");
    m_benchmarkAction->setShortcut(QKeySequence("Ctrl+B"));
    connect(m_benchmarkAction, &QAction::triggered, [this]() {
        m_tabWidget->setCurrentIndex(3); // Benchmark tab
    });
    
    m_chatAction = toolsMenu->addAction("&Open Chat");
    m_chatAction->setShortcut(QKeySequence("Ctrl+C"));
    connect(m_chatAction, &QAction::triggered, [this]() {
        m_tabWidget->setCurrentIndex(0); // Chat tab
    });
    
    toolsMenu->addSeparator();
    
    QAction *settingsAction = toolsMenu->addAction("&Settings");
    connect(settingsAction, &QAction::triggered, this, &MainWindow::showSettings);
    
    // Help menu
    QMenu *helpMenu = menuBar()->addMenu("&Help");
    
    QAction *aboutAction = helpMenu->addAction("&About");
    connect(aboutAction, &QAction::triggered, this, &MainWindow::showAbout);
}

void MainWindow::setupToolBar()
{
    QToolBar *toolBar = addToolBar("Main Toolbar");
    toolBar->setMovable(false);
    
    // Model actions
    toolBar->addAction(m_openModelAction);
    toolBar->addSeparator();
    
    // Quick access actions
    toolBar->addAction(m_chatAction);
    toolBar->addAction(m_benchmarkAction);
    toolBar->addSeparator();
    
    // Results action
    toolBar->addAction(m_saveResultsAction);
}

void MainWindow::setupStatusBar()
{
    // Status label
    m_statusLabel = new QLabel("Ready", this);
    statusBar()->addWidget(m_statusLabel);
    
    // Model info
    m_modelLabel = new QLabel("No model loaded", this);
    m_modelLabel->setStyleSheet("QLabel { color: #dc3545; font-weight: bold; }");
    statusBar()->addPermanentWidget(m_modelLabel);
    
    // Performance indicators
    m_cpuLabel = new QLabel("CPU: --", this);
    m_cpuLabel->setMinimumWidth(80);
    statusBar()->addPermanentWidget(m_cpuLabel);
    
    m_memoryLabel = new QLabel("RAM: --", this);
    m_memoryLabel->setMinimumWidth(100);
    statusBar()->addPermanentWidget(m_memoryLabel);
    
    // Progress bar
    m_progressBar = new QProgressBar(this);
    m_progressBar->setVisible(false);
    m_progressBar->setMaximumWidth(200);
    statusBar()->addPermanentWidget(m_progressBar);
}

void MainWindow::loadSettings()
{
    // Load window geometry
    restoreGeometry(m_settings.value("geometry").toByteArray());
    restoreState(m_settings.value("windowState").toByteArray());
    
    // Load recent files
    // TODO: Implement recent files functionality
}

void MainWindow::saveSettings()
{
    // Save window geometry
    m_settings.setValue("geometry", saveGeometry());
    m_settings.setValue("windowState", saveState());
}

void MainWindow::onModelLoaded(const QString &modelPath)
{
    m_statusLabel->setText(QString("Model loaded: %1").arg(QFileInfo(modelPath).fileName()));
    m_progressBar->setVisible(false);
    updateModelInfo();
    
    // Enable actions
    m_benchmarkAction->setEnabled(true);
    m_chatAction->setEnabled(true);
}

void MainWindow::onCompressionCompleted(const CompressionResult &result)
{
    if (result.success) {
        m_statusLabel->setText(QString("Compression completed: %1x ratio").arg(result.compressionRatio, 0, 'f', 2));
        m_resultsWidget->addCompressionResult(result);
    } else {
        m_statusLabel->setText("Compression failed");
        QMessageBox::warning(this, "Compression Error", result.errorMessage);
    }
    m_progressBar->setVisible(false);
}

void MainWindow::onInferenceCompleted(const InferenceResult &result)
{
    if (result.success) {
        m_statusLabel->setText(QString("Inference completed in %1 ms").arg(result.inferenceTimeMs, 0, 'f', 1));
        m_resultsWidget->addInferenceResult(result);
    } else {
        m_statusLabel->setText("Inference failed");
        QMessageBox::warning(this, "Inference Error", result.errorMessage);
    }
    m_progressBar->setVisible(false);
}

void MainWindow::onBenchmarkCompleted(const BenchmarkResult &result)
{
    m_statusLabel->setText(QString("Benchmark completed: %1 runs").arg(result.numRuns));
    m_resultsWidget->addBenchmarkResult(result);
    m_progressBar->setVisible(false);
}

void MainWindow::onPerformanceUpdate(const PerformanceMetrics &metrics)
{
    m_cpuLabel->setText(QString("CPU: %1%").arg(metrics.cpuUsage, 0, 'f', 1));
    m_memoryLabel->setText(QString("RAM: %1 MB").arg(metrics.memoryUsageMB, 0, 'f', 1));
}

void MainWindow::openModel()
{
    QString modelPath = QFileDialog::getOpenFileName(
        this,
        "Open AI Model",
        QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation),
        "Model Files (*.onnx *.pt *.pth *.pb *.tflite *.sdr);;All Files (*.*)"
    );
    
    if (!modelPath.isEmpty()) {
        m_progressBar->setVisible(true);
        m_progressBar->setRange(0, 0); // Indeterminate progress
        m_statusLabel->setText("Loading model...");
        
        // Load model in background
        QtConcurrent::run([this, modelPath]() {
            m_modelManager->loadModel(modelPath);
        });
    }
}

void MainWindow::saveResults()
{
    QString fileName = QFileDialog::getSaveFileName(
        this,
        "Save Results",
        QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation) + "/cortexsdr_results.json",
        "JSON Files (*.json);;Text Files (*.txt);;All Files (*.*)"
    );
    
    if (!fileName.isEmpty()) {
        // TODO: Implement results saving
        m_statusLabel->setText("Results saved successfully");
    }
}

void MainWindow::showAbout()
{
    QMessageBox::about(this, "About CortexSDR Desktop",
        "<h3>CortexSDR Desktop</h3>"
        "<p>AI Model Compression & Inference Tool</p>"
        "<p>Version 1.0.0</p>"
        "<p>Built with Qt6 and CortexSDR SDK</p>"
        "<p>Features:</p>"
        "<ul>"
        "<li>💬 Interactive Chat Interface</li>"
        "<li>🗜️ Model Compression</li>"
        "<li>⚡ Fast Inference</li>"
        "<li>📊 Performance Benchmarking</li>"
        "<li>📋 Results Management</li>"
        "</ul>");
}

void MainWindow::showSettings()
{
    // TODO: Implement settings dialog
    QMessageBox::information(this, "Settings", "Settings dialog will be implemented in a future version.");
}

void MainWindow::showModelManager()
{
    // TODO: Implement model manager dialog
    QMessageBox::information(this, "Model Manager", "Model manager will be implemented in a future version.");
}

void MainWindow::updateModelInfo()
{
    if (m_modelManager && m_modelManager->isModelLoaded()) {
        QString modelPath = m_modelManager->getCurrentModelPath();
        QFileInfo fileInfo(modelPath);
        m_modelLabel->setText(QString("Model: %1").arg(fileInfo.fileName()));
        m_modelLabel->setStyleSheet("QLabel { color: #28a745; font-weight: bold; }");
    } else {
        m_modelLabel->setText("No model loaded");
        m_modelLabel->setStyleSheet("QLabel { color: #dc3545; font-weight: bold; }");
    }
} 