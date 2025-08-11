#include "TextGenerationWidget.h"

TextGenerationWidget::TextGenerationWidget(ModelManager *modelManager, QWidget *parent)
    : QWidget(parent)
    , m_modelManager(modelManager)
    , m_generating(false)
{
    setupUI();
}

void TextGenerationWidget::setupUI()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    
    // Input Group
    m_inputGroup = new QGroupBox("Input Text", this);
    QVBoxLayout *inputLayout = new QVBoxLayout(m_inputGroup);
    
    m_inputLabel = new QLabel("Enter your prompt:", this);
    inputLayout->addWidget(m_inputLabel);
    
    m_inputTextEdit = new QTextEdit(this);
    m_inputTextEdit->setMaximumHeight(100);
    m_inputTextEdit->setPlaceholderText("Enter your text prompt here...");
    inputLayout->addWidget(m_inputTextEdit);
    
    mainLayout->addWidget(m_inputGroup);
    
    // Settings Group
    m_settingsGroup = new QGroupBox("Generation Settings", this);
    QGridLayout *settingsLayout = new QGridLayout(m_settingsGroup);
    
    // Max Length
    settingsLayout->addWidget(new QLabel("Max Length:"), 0, 0);
    m_maxLengthSlider = new QSlider(Qt::Horizontal, this);
    m_maxLengthSlider->setRange(10, 1000);
    m_maxLengthSlider->setValue(100);
    settingsLayout->addWidget(m_maxLengthSlider, 0, 1);
    
    m_maxLengthSpinBox = new QSpinBox(this);
    m_maxLengthSpinBox->setRange(10, 1000);
    m_maxLengthSpinBox->setValue(100);
    settingsLayout->addWidget(m_maxLengthSpinBox, 0, 2);
    
    // Temperature
    settingsLayout->addWidget(new QLabel("Temperature:"), 1, 0);
    m_temperatureSlider = new QSlider(Qt::Horizontal, this);
    m_temperatureSlider->setRange(0, 200);
    m_temperatureSlider->setValue(70);
    settingsLayout->addWidget(m_temperatureSlider, 1, 1);
    
    m_temperatureSpinBox = new QDoubleSpinBox(this);
    m_temperatureSpinBox->setRange(0.0, 2.0);
    m_temperatureSpinBox->setSingleStep(0.1);
    m_temperatureSpinBox->setValue(0.7);
    settingsLayout->addWidget(m_temperatureSpinBox, 1, 2);
    
    // Top P
    settingsLayout->addWidget(new QLabel("Top P:"), 2, 0);
    m_topPSlider = new QSlider(Qt::Horizontal, this);
    m_topPSlider->setRange(0, 100);
    m_topPSlider->setValue(90);
    settingsLayout->addWidget(m_topPSlider, 2, 1);
    
    m_topPSpinBox = new QDoubleSpinBox(this);
    m_topPSpinBox->setRange(0.0, 1.0);
    m_topPSpinBox->setSingleStep(0.01);
    m_topPSpinBox->setValue(0.9);
    settingsLayout->addWidget(m_topPSpinBox, 2, 2);
    
    // Beam Search
    m_useBeamSearchCheck = new QCheckBox("Use Beam Search", this);
    settingsLayout->addWidget(m_useBeamSearchCheck, 3, 0);
    
    settingsLayout->addWidget(new QLabel("Beam Width:"), 3, 1);
    m_beamWidthSpinBox = new QSpinBox(this);
    m_beamWidthSpinBox->setRange(1, 10);
    m_beamWidthSpinBox->setValue(5);
    settingsLayout->addWidget(m_beamWidthSpinBox, 3, 2);
    
    mainLayout->addWidget(m_settingsGroup);
    
    // Output Group
    m_outputGroup = new QGroupBox("Generated Text", this);
    QVBoxLayout *outputLayout = new QVBoxLayout(m_outputGroup);
    
    m_outputLabel = new QLabel("Generated output:", this);
    outputLayout->addWidget(m_outputLabel);
    
    m_outputTextEdit = new QTextEdit(this);
    m_outputTextEdit->setReadOnly(true);
    m_outputTextEdit->setPlaceholderText("Generated text will appear here...");
    outputLayout->addWidget(m_outputTextEdit);
    
    mainLayout->addWidget(m_outputGroup);
    
    // Actions
    QHBoxLayout *actionsLayout = new QHBoxLayout();
    
    m_generateButton = new QPushButton("Generate Text", this);
    connect(m_generateButton, &QPushButton::clicked, this, &TextGenerationWidget::onGenerate);
    actionsLayout->addWidget(m_generateButton);
    
    m_clearButton = new QPushButton("Clear", this);
    connect(m_clearButton, &QPushButton::clicked, this, &TextGenerationWidget::onClear);
    actionsLayout->addWidget(m_clearButton);
    
    actionsLayout->addStretch();
    mainLayout->addLayout(actionsLayout);
    
    // Connect signals
    connect(m_maxLengthSlider, &QSlider::valueChanged, this, &TextGenerationWidget::onMaxLengthChanged);
    connect(m_maxLengthSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &TextGenerationWidget::onMaxLengthChanged);
    connect(m_temperatureSlider, &QSlider::valueChanged, this, &TextGenerationWidget::onTemperatureChanged);
    connect(m_temperatureSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &TextGenerationWidget::onTemperatureChanged);
    connect(m_topPSlider, &QSlider::valueChanged, this, &TextGenerationWidget::onTopPChanged);
    connect(m_topPSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &TextGenerationWidget::onTopPChanged);
}

void TextGenerationWidget::onGenerate()
{
    if (!m_modelManager || m_generating) return;
    
    QString inputText = m_inputTextEdit->toPlainText();
    if (inputText.isEmpty()) {
        return;
    }
    
    int maxLength = m_maxLengthSpinBox->value();
    m_modelManager->runTextInference(inputText, maxLength);
}

void TextGenerationWidget::onClear()
{
    m_inputTextEdit->clear();
    m_outputTextEdit->clear();
}

void TextGenerationWidget::onMaxLengthChanged(int value)
{
    if (sender() == m_maxLengthSlider) {
        m_maxLengthSpinBox->setValue(value);
    } else if (sender() == m_maxLengthSpinBox) {
        m_maxLengthSlider->setValue(value);
    }
}

void TextGenerationWidget::onTemperatureChanged(double value)
{
    if (sender() == m_temperatureSlider) {
        m_temperatureSpinBox->setValue(value);
    } else if (sender() == m_temperatureSpinBox) {
        m_temperatureSlider->setValue(static_cast<int>(value * 100));
    }
}

void TextGenerationWidget::onTopPChanged(double value)
{
    if (sender() == m_topPSlider) {
        m_topPSpinBox->setValue(value);
    } else if (sender() == m_topPSpinBox) {
        m_topPSlider->setValue(static_cast<int>(value * 100));
    }
}

void TextGenerationWidget::enableControls(bool enable)
{
    m_generateButton->setEnabled(enable);
    m_clearButton->setEnabled(enable);
    m_inputTextEdit->setEnabled(enable);
} 