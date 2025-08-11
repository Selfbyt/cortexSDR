#include "InferenceWidget.h"

InferenceWidget::InferenceWidget(ModelManager *modelManager, QWidget *parent)
    : QWidget(parent)
    , m_modelManager(modelManager)
    , m_inferencing(false)
{
    setupUI();
    
    if (m_modelManager) {
        connect(m_modelManager, &ModelManager::inferenceCompleted, this, &InferenceWidget::onInferenceCompleted);
        connect(m_modelManager, &ModelManager::modelLoaded, this, &InferenceWidget::onModelLoaded);
    }
}

void InferenceWidget::setupUI()
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
    modelLayout->addWidget(m_browseModelButton, 0, 2);
    
    modelLayout->addWidget(new QLabel("Format:"), 1, 0);
    m_modelFormatLabel = new QLabel("--", this);
    modelLayout->addWidget(m_modelFormatLabel, 1, 1);
    
    modelLayout->addWidget(new QLabel("Size:"), 2, 0);
    m_modelSizeLabel = new QLabel("--", this);
    modelLayout->addWidget(m_modelSizeLabel, 2, 1);
    
    modelLayout->addWidget(new QLabel("Type:"), 3, 0);
    m_modelTypeCombo = new QComboBox(this);
    m_modelTypeCombo->addItems({"Text Generation", "Audio Processing", "Image Classification"});
    modelLayout->addWidget(m_modelTypeCombo, 3, 1);
    
    mainLayout->addWidget(m_modelGroup);
    
    // Inference Tabs
    m_inferenceTabs = new QTabWidget(this);
    
    // Text Generation Tab
    m_textWidget = new TextGenerationWidget(m_modelManager, this);
    m_inferenceTabs->addTab(m_textWidget, "Text Generation");
    
    // Audio Inference Tab
    m_audioWidget = new AudioInferenceWidget(m_modelManager, this);
    m_inferenceTabs->addTab(m_audioWidget, "Audio Processing");
    
    mainLayout->addWidget(m_inferenceTabs);
    
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
}

void InferenceWidget::onModelLoaded(const QString &modelPath)
{
    m_currentModelPath = modelPath;
    updateModelInfo();
}

void InferenceWidget::onInferenceCompleted(const InferenceResult &result)
{
    m_inferencing = false;
    m_progressBar->setVisible(false);
    enableControls(true);
    
    if (result.success) {
        m_statusLabel->setText("Inference completed successfully");
        m_resultsText->setText(result.outputText);
    } else {
        m_statusLabel->setText("Inference failed");
        m_resultsText->setText(QString("Error: %1").arg(result.errorMessage));
    }
}

void InferenceWidget::onInferenceStarted()
{
    m_inferencing = true;
    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, 0);
    m_statusLabel->setText("Running inference...");
    enableControls(false);
}

void InferenceWidget::onModelTypeChanged(int index)
{
    m_currentModelType = m_modelTypeCombo->itemText(index);
}

void InferenceWidget::onLoadCompressedModel()
{
    // TODO: Implement loading compressed model
}



void InferenceWidget::updateModelInfo()
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

void InferenceWidget::enableControls(bool enable)
{
    m_browseModelButton->setEnabled(enable);
    m_modelTypeCombo->setEnabled(enable);
} 