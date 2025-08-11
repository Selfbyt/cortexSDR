#include "AudioInferenceWidget.h"

AudioInferenceWidget::AudioInferenceWidget(ModelManager *modelManager, QWidget *parent)
    : QWidget(parent)
    , m_modelManager(modelManager)
    , m_processing(false)
{
    setupUI();
}

void AudioInferenceWidget::setupUI()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    
    // Audio Input Group
    m_audioGroup = new QGroupBox("Audio Input", this);
    QGridLayout *audioLayout = new QGridLayout(m_audioGroup);
    
    audioLayout->addWidget(new QLabel("Audio File:"), 0, 0);
    m_audioPathEdit = new QLineEdit(this);
    m_audioPathEdit->setReadOnly(true);
    audioLayout->addWidget(m_audioPathEdit, 0, 1);
    
    m_browseAudioButton = new QPushButton("Browse", this);
    connect(m_browseAudioButton, &QPushButton::clicked, this, &AudioInferenceWidget::onBrowseAudio);
    audioLayout->addWidget(m_browseAudioButton, 0, 2);
    
    audioLayout->addWidget(new QLabel("Format:"), 1, 0);
    m_audioFormatLabel = new QLabel("--", this);
    audioLayout->addWidget(m_audioFormatLabel, 1, 1);
    
    audioLayout->addWidget(new QLabel("Duration:"), 2, 0);
    m_audioDurationLabel = new QLabel("--", this);
    audioLayout->addWidget(m_audioDurationLabel, 2, 1);
    
    audioLayout->addWidget(new QLabel("Size:"), 3, 0);
    m_audioSizeLabel = new QLabel("--", this);
    audioLayout->addWidget(m_audioSizeLabel, 3, 1);
    
    mainLayout->addWidget(m_audioGroup);
    
    // Settings Group
    m_settingsGroup = new QGroupBox("Audio Settings", this);
    QGridLayout *settingsLayout = new QGridLayout(m_settingsGroup);
    
    settingsLayout->addWidget(new QLabel("Sample Rate:"), 0, 0);
    m_sampleRateCombo = new QComboBox(this);
    m_sampleRateCombo->addItems({"8000 Hz", "16000 Hz", "22050 Hz", "44100 Hz", "48000 Hz"});
    m_sampleRateCombo->setCurrentText("16000 Hz");
    settingsLayout->addWidget(m_sampleRateCombo, 0, 1);
    
    settingsLayout->addWidget(new QLabel("Channels:"), 1, 0);
    m_channelsCombo = new QComboBox(this);
    m_channelsCombo->addItems({"Mono", "Stereo"});
    settingsLayout->addWidget(m_channelsCombo, 1, 1);
    
    settingsLayout->addWidget(new QLabel("Format:"), 2, 0);
    m_audioFormatCombo = new QComboBox(this);
    m_audioFormatCombo->addItems({"WAV", "MP3", "FLAC", "OGG"});
    settingsLayout->addWidget(m_audioFormatCombo, 2, 1);
    
    settingsLayout->addWidget(new QLabel("Volume:"), 3, 0);
    m_volumeSlider = new QSlider(Qt::Horizontal, this);
    m_volumeSlider->setRange(0, 100);
    m_volumeSlider->setValue(100);
    settingsLayout->addWidget(m_volumeSlider, 3, 1);
    
    m_volumeSpinBox = new QDoubleSpinBox(this);
    m_volumeSpinBox->setRange(0.0, 2.0);
    m_volumeSpinBox->setSingleStep(0.1);
    m_volumeSpinBox->setValue(1.0);
    settingsLayout->addWidget(m_volumeSpinBox, 3, 2);
    
    m_normalizeCheck = new QCheckBox("Normalize Audio", this);
    m_normalizeCheck->setChecked(true);
    settingsLayout->addWidget(m_normalizeCheck, 4, 0);
    
    m_removeNoiseCheck = new QCheckBox("Remove Noise", this);
    settingsLayout->addWidget(m_removeNoiseCheck, 4, 1);
    
    mainLayout->addWidget(m_settingsGroup);
    
    // Output Group
    m_outputGroup = new QGroupBox("Output", this);
    QVBoxLayout *outputLayout = new QVBoxLayout(m_outputGroup);
    
    m_outputLabel = new QLabel("Processing Results:", this);
    outputLayout->addWidget(m_outputLabel);
    
    m_outputTextEdit = new QTextEdit(this);
    m_outputTextEdit->setReadOnly(true);
    m_outputTextEdit->setMaximumHeight(100);
    outputLayout->addWidget(m_outputTextEdit);
    
    QHBoxLayout *outputAudioLayout = new QHBoxLayout();
    outputAudioLayout->addWidget(new QLabel("Output Audio:"));
    m_outputAudioPathEdit = new QLineEdit(this);
    outputAudioLayout->addWidget(m_outputAudioPathEdit);
    
    m_browseOutputAudioButton = new QPushButton("Browse", this);
    outputAudioLayout->addWidget(m_browseOutputAudioButton);
    
    m_playOutputButton = new QPushButton("Play", this);
    m_playOutputButton->setEnabled(false);
    outputAudioLayout->addWidget(m_playOutputButton);
    
    outputLayout->addLayout(outputAudioLayout);
    
    mainLayout->addWidget(m_outputGroup);
    
    // Actions
    QHBoxLayout *actionsLayout = new QHBoxLayout();
    
    m_processButton = new QPushButton("Process Audio", this);
    m_processButton->setEnabled(false);
    connect(m_processButton, &QPushButton::clicked, this, &AudioInferenceWidget::onProcessAudio);
    actionsLayout->addWidget(m_processButton);
    
    m_clearButton = new QPushButton("Clear", this);
    connect(m_clearButton, &QPushButton::clicked, this, &AudioInferenceWidget::onClear);
    actionsLayout->addWidget(m_clearButton);
    
    actionsLayout->addStretch();
    mainLayout->addLayout(actionsLayout);
    
    // Connect signals
    connect(m_sampleRateCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &AudioInferenceWidget::onSampleRateChanged);
    connect(m_channelsCombo, QOverload<int>::of(&QComboBox::currentIndexChanged), 
            this, &AudioInferenceWidget::onChannelsChanged);
}

void AudioInferenceWidget::onBrowseAudio()
{
    QString audioPath = QFileDialog::getOpenFileName(
        this,
        "Select Audio File",
        QStandardPaths::writableLocation(QStandardPaths::MusicLocation),
        "Audio Files (*.wav *.mp3 *.flac *.ogg *.m4a);;All Files (*.*)"
    );
    
    if (!audioPath.isEmpty()) {
        m_audioPathEdit->setText(audioPath);
        m_currentAudioPath = audioPath;
        updateAudioInfo();
        m_processButton->setEnabled(true);
    }
}

void AudioInferenceWidget::onProcessAudio()
{
    if (!m_modelManager || m_processing || m_currentAudioPath.isEmpty()) return;
    
    m_modelManager->runAudioInference(m_currentAudioPath);
}

void AudioInferenceWidget::onClear()
{
    m_audioPathEdit->clear();
    m_outputTextEdit->clear();
    m_outputAudioPathEdit->clear();
    m_audioFormatLabel->setText("--");
    m_audioDurationLabel->setText("--");
    m_audioSizeLabel->setText("--");
    m_currentAudioPath.clear();
    m_processButton->setEnabled(false);
    m_playOutputButton->setEnabled(false);
}

void AudioInferenceWidget::onSampleRateChanged(int value)
{
    // Handle sample rate change
}

void AudioInferenceWidget::onChannelsChanged(int value)
{
    // Handle channels change
}



void AudioInferenceWidget::enableControls(bool enable)
{
    m_browseAudioButton->setEnabled(enable);
    m_processButton->setEnabled(enable && !m_currentAudioPath.isEmpty());
    m_clearButton->setEnabled(enable);
    m_sampleRateCombo->setEnabled(enable);
    m_channelsCombo->setEnabled(enable);
    m_audioFormatCombo->setEnabled(enable);
    m_volumeSlider->setEnabled(enable);
    m_volumeSpinBox->setEnabled(enable);
    m_normalizeCheck->setEnabled(enable);
    m_removeNoiseCheck->setEnabled(enable);
}

void AudioInferenceWidget::updateAudioInfo()
{
    if (m_currentAudioPath.isEmpty()) return;
    
    QFileInfo fileInfo(m_currentAudioPath);
    m_audioFormatLabel->setText(fileInfo.suffix().toUpper());
    
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
    m_audioSizeLabel->setText(sizeStr);
    
    // TODO: Get actual duration from audio file
    m_audioDurationLabel->setText("Unknown");
} 