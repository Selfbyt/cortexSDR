#include "CompressionWidget.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QStandardPaths>

CompressionWidget::CompressionWidget(ModelManager *modelManager, QWidget *parent)
    : QWidget(parent)
    , m_modelManager(modelManager)
    , m_compressing(false)
{
    setupUI();
    
    // Connect signals
    if (m_modelManager) {
        connect(m_modelManager, &ModelManager::compressionStarted, this, &CompressionWidget::onCompressionStarted);
        connect(m_modelManager, &ModelManager::compressionProgress, this, &CompressionWidget::onCompressionProgress);
        connect(m_modelManager, &ModelManager::compressionCompleted, this, &CompressionWidget::onCompressionCompleted);
        connect(m_modelManager, &ModelManager::modelLoaded, this, &CompressionWidget::onModelLoaded);
    }
}

void CompressionWidget::setupUI()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    
    // Model Selection Group
    m_modelGroup = new QGroupBox("Model Selection", this);
    QGridLayout *modelLayout = new QGridLayout(m_modelGroup);
    
    modelLayout->addWidget(new QLabel("Model Path:"), 0, 0);
    m_modelPathEdit = new QLineEdit(this);
    m_modelPathEdit->setReadOnly(true);
    modelLayout->addWidget(m_modelPathEdit, 0, 1);
    
    m_browseModelButton = new QPushButton("Browse", this);
    connect(m_browseModelButton, &QPushButton::clicked, this, &CompressionWidget::onBrowseModel);
    modelLayout->addWidget(m_browseModelButton, 0, 2);
    
    modelLayout->addWidget(new QLabel("Format:"), 1, 0);
    m_modelFormatLabel = new QLabel("--", this);
    modelLayout->addWidget(m_modelFormatLabel, 1, 1);
    
    modelLayout->addWidget(new QLabel("Size:"), 2, 0);
    m_modelSizeLabel = new QLabel("--", this);
    modelLayout->addWidget(m_modelSizeLabel, 2, 1);
    
    mainLayout->addWidget(m_modelGroup);
    
    // Output Group
    m_outputGroup = new QGroupBox("Output", this);
    QHBoxLayout *outputLayout = new QHBoxLayout(m_outputGroup);
    
    outputLayout->addWidget(new QLabel("Output Path:"));
    m_outputPathEdit = new QLineEdit(this);
    outputLayout->addWidget(m_outputPathEdit);
    
    m_browseOutputButton = new QPushButton("Browse", this);
    connect(m_browseOutputButton, &QPushButton::clicked, this, &CompressionWidget::onBrowseOutput);
    outputLayout->addWidget(m_browseOutputButton);
    
    mainLayout->addWidget(m_outputGroup);
    
    // Settings Group
    m_settingsGroup = new QGroupBox("Compression Settings", this);
    QGridLayout *settingsLayout = new QGridLayout(m_settingsGroup);
    
    // Sparsity
    settingsLayout->addWidget(new QLabel("Sparsity:"), 0, 0);
    m_sparsitySlider = new QSlider(Qt::Horizontal, this);
    m_sparsitySlider->setRange(1, 50); // 1% to 50%
    m_sparsitySlider->setValue(2); // Default 2%
    connect(m_sparsitySlider, &QSlider::valueChanged, this, &CompressionWidget::onSparsityChanged);
    settingsLayout->addWidget(m_sparsitySlider, 0, 1);
    
    m_sparsitySpinBox = new QDoubleSpinBox(this);
    m_sparsitySpinBox->setRange(0.01, 0.5);
    m_sparsitySpinBox->setSingleStep(0.01);
    m_sparsitySpinBox->setValue(0.02);
    m_sparsitySpinBox->setSuffix("%");
    connect(m_sparsitySpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), 
            this, &CompressionWidget::onSparsityChanged);
    settingsLayout->addWidget(m_sparsitySpinBox, 0, 2);
    
    // Compression Level
    settingsLayout->addWidget(new QLabel("Compression Level:"), 1, 0);
    m_compressionLevelSlider = new QSlider(Qt::Horizontal, this);
    m_compressionLevelSlider->setRange(1, 9);
    m_compressionLevelSlider->setValue(6);
    connect(m_compressionLevelSlider, &QSlider::valueChanged, this, &CompressionWidget::onCompressionLevelChanged);
    settingsLayout->addWidget(m_compressionLevelSlider, 1, 1);
    
    m_compressionLevelSpinBox = new QSpinBox(this);
    m_compressionLevelSpinBox->setRange(1, 9);
    m_compressionLevelSpinBox->setValue(6);
    connect(m_compressionLevelSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), 
            this, &CompressionWidget::onCompressionLevelChanged);
    settingsLayout->addWidget(m_compressionLevelSpinBox, 1, 2);
    
    // Quantization
    m_useQuantizationCheck = new QCheckBox("Use Quantization", this);
    settingsLayout->addWidget(m_useQuantizationCheck, 2, 0);
    
    settingsLayout->addWidget(new QLabel("Quantization Bits:"), 2, 1);
    m_quantizationBitsSpinBox = new QSpinBox(this);
    m_quantizationBitsSpinBox->setRange(4, 16);
    m_quantizationBitsSpinBox->setValue(8);
    settingsLayout->addWidget(m_quantizationBitsSpinBox, 2, 2);
    
    mainLayout->addWidget(m_settingsGroup);
    
    // Progress Group
    m_progressGroup = new QGroupBox("Progress", this);
    QVBoxLayout *progressLayout = new QVBoxLayout(m_progressGroup);
    
    m_progressBar = new QProgressBar(this);
    m_progressBar->setVisible(false);
    progressLayout->addWidget(m_progressBar);
    
    m_statusLabel = new QLabel("Ready", this);
    progressLayout->addWidget(m_statusLabel);
    
    mainLayout->addWidget(m_progressGroup);
    
    // Results Group
    m_resultsGroup = new QGroupBox("Results", this);
    QVBoxLayout *resultsLayout = new QVBoxLayout(m_resultsGroup);
    
    m_resultsText = new QTextEdit(this);
    m_resultsText->setReadOnly(true);
    m_resultsText->setMaximumHeight(150);
    resultsLayout->addWidget(m_resultsText);
    
    mainLayout->addWidget(m_resultsGroup);
    
    // Actions
    QHBoxLayout *actionsLayout = new QHBoxLayout();
    
    m_compressButton = new QPushButton("Compress Model", this);
    m_compressButton->setEnabled(false);
    connect(m_compressButton, &QPushButton::clicked, this, &CompressionWidget::onCompress);
    actionsLayout->addWidget(m_compressButton);
    
    m_clearButton = new QPushButton("Clear", this);
    connect(m_clearButton, &QPushButton::clicked, this, &CompressionWidget::onClear);
    actionsLayout->addWidget(m_clearButton);
    
    actionsLayout->addStretch();
    mainLayout->addLayout(actionsLayout);
}

void CompressionWidget::onBrowseModel()
{
    QString modelPath = QFileDialog::getOpenFileName(
        this,
        "Select Model File",
        QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation),
        "Model Files (*.onnx *.gguf *.pt *.pth *.pb *.h5);;All Files (*.*)"
    );
    
    if (!modelPath.isEmpty()) {
        m_modelPathEdit->setText(modelPath);
        if (m_modelManager) {
            m_modelManager->loadModel(modelPath);
        }
    }
}

void CompressionWidget::onBrowseOutput()
{
    QString outputPath = QFileDialog::getSaveFileName(
        this,
        "Save Compressed Model",
        QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation),
        "SDR Files (*.sdr);;All Files (*.*)"
    );
    
    if (!outputPath.isEmpty()) {
        m_outputPathEdit->setText(outputPath);
    }
}

void CompressionWidget::onCompress()
{
    if (!m_modelManager || m_compressing) return;
    
    QString outputPath = m_outputPathEdit->text();
    if (outputPath.isEmpty()) {
        QMessageBox::warning(this, "Error", "Please select an output path");
        return;
    }
    
    float sparsity = m_sparsitySpinBox->value() / 100.0f;
    int compressionLevel = m_compressionLevelSpinBox->value();
    bool useQuantization = m_useQuantizationCheck->isChecked();
    int quantizationBits = m_quantizationBitsSpinBox->value();
    
    m_modelManager->compressModel(outputPath, sparsity, compressionLevel, useQuantization, quantizationBits);
}

void CompressionWidget::onCompressionStarted()
{
    m_compressing = true;
    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, 0); // Indeterminate
    m_statusLabel->setText("Compressing...");
    enableControls(false);
}

void CompressionWidget::onCompressionProgress(int percentage)
{
    m_progressBar->setRange(0, 100);
    m_progressBar->setValue(percentage);
    m_statusLabel->setText(QString("Compressing... %1%").arg(percentage));
}

void CompressionWidget::onCompressionCompleted(const CompressionResult &result)
{
    m_compressing = false;
    m_progressBar->setVisible(false);
    enableControls(true);
    
    if (result.success) {
        m_statusLabel->setText("Compression completed successfully");
        
        QString resultsText = QString(
            "Compression Results:\n"
            "Original Size: %1 bytes\n"
            "Compressed Size: %2 bytes\n"
            "Compression Ratio: %3x\n"
            "Compression Time: %4 ms\n"
        ).arg(result.originalSize)
         .arg(result.compressedSize)
         .arg(result.compressionRatio, 0, 'f', 2)
         .arg(result.compressionTimeMs, 0, 'f', 1);
        
        m_resultsText->setText(resultsText);
    } else {
        m_statusLabel->setText("Compression failed");
        m_resultsText->setText(QString("Error: %1").arg(result.errorMessage));
    }
}

void CompressionWidget::onModelLoaded(const QString &modelPath)
{
    m_currentModelPath = modelPath;
    updateModelInfo();
    m_compressButton->setEnabled(true);
}

void CompressionWidget::onSparsityChanged(double value)
{
    // Sync slider and spinbox
    if (sender() == m_sparsitySlider) {
        m_sparsitySpinBox->setValue(value);
    } else if (sender() == m_sparsitySpinBox) {
        m_sparsitySlider->setValue(static_cast<int>(value));
    }
}

void CompressionWidget::onCompressionLevelChanged(int value)
{
    // Sync slider and spinbox
    if (sender() == m_compressionLevelSlider) {
        m_compressionLevelSpinBox->setValue(value);
    } else if (sender() == m_compressionLevelSpinBox) {
        m_compressionLevelSlider->setValue(value);
    }
}

void CompressionWidget::updateModelInfo()
{
    if (m_currentModelPath.isEmpty()) return;
    
    QFileInfo fileInfo(m_currentModelPath);
    m_modelFormatLabel->setText(fileInfo.suffix().toUpper());
    
    qint64 size = fileInfo.size();
    QString sizeStr;
    if (size < 1024) {
        sizeStr = QString("%1 B").arg(size);
    } else if (size < 1024 * 1024) {
        sizeStr = QString("%1 KB").arg(size / 1024.0, 0, 'f', 1);
    } else if (size < 1024 * 1024 * 1024) {
        sizeStr = QString("%1 MB").arg(size / (1024.0 * 1024.0), 0, 'f', 1);
    } else {
        sizeStr = QString("%1 GB").arg(size / (1024.0 * 1024.0 * 1024.0), 0, 'f', 1);
    }
    m_modelSizeLabel->setText(sizeStr);
}

void CompressionWidget::enableControls(bool enable)
{
    m_browseModelButton->setEnabled(enable);
    m_browseOutputButton->setEnabled(enable);
    m_compressButton->setEnabled(enable && !m_currentModelPath.isEmpty());
    m_clearButton->setEnabled(enable);
    
    // Settings
    m_sparsitySlider->setEnabled(enable);
    m_sparsitySpinBox->setEnabled(enable);
    m_compressionLevelSlider->setEnabled(enable);
    m_compressionLevelSpinBox->setEnabled(enable);
    m_useQuantizationCheck->setEnabled(enable);
    m_quantizationBitsSpinBox->setEnabled(enable);
}

void CompressionWidget::onClear()
{
    m_modelPathEdit->clear();
    m_outputPathEdit->clear();
    m_resultsText->clear();
    m_modelFormatLabel->setText("--");
    m_modelSizeLabel->setText("--");
    m_currentModelPath.clear();
    m_compressButton->setEnabled(false);
    m_statusLabel->setText("Ready");
} 