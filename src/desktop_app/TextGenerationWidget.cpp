#include "TextGenerationWidget.h"
#include <QApplication>
#include <QMessageBox>
#include <QStandardPaths>
#include <QDir>
#include <QFileInfo>
#include <QTimer>
#include <QRandomGenerator>
#include <QDateTime>
#include <QScrollBar>

// MessageBubble implementation
MessageBubble::MessageBubble(const QString &text, bool isUser, QWidget *parent)
    : QFrame(parent)
    , m_text(text)
    , m_isUser(isUser)
{
    setupUI();
}

void MessageBubble::setupUI()
{
    QHBoxLayout *layout = new QHBoxLayout(this);
    layout->setContentsMargins(8, 4, 8, 4);
    
    if (m_isUser) {
        layout->addStretch();
        layout->addWidget(new QLabel(m_text, this));
        setStyleSheet("QFrame { background-color: #007AFF; border-radius: 18px; padding: 8px; } QLabel { color: white; font-size: 14px; }");
    } else {
        layout->addWidget(new QLabel(m_text, this));
        layout->addStretch();
        setStyleSheet("QFrame { background-color: #F0F0F0; border-radius: 18px; padding: 8px; } QLabel { color: black; font-size: 14px; }");
    }
}

TextGenerationWidget::TextGenerationWidget(ModelManager *modelManager, QWidget *parent)
    : QWidget(parent)
    , m_modelManager(modelManager)
    , m_generating(false)
    , m_modelLoaded(false)
    , m_isRecording(false)
    , m_settingsVisible(false)
{
    setupUI();
    
    if (m_modelManager) {
        connect(m_modelManager, &ModelManager::modelLoaded, this, &TextGenerationWidget::onModelLoaded);
        connect(m_modelManager, &ModelManager::modelLoadFailed, this, &TextGenerationWidget::onModelLoadFailed);
        connect(m_modelManager, &ModelManager::inferenceCompleted, this, &TextGenerationWidget::onInferenceCompleted);
        connect(m_modelManager, &ModelManager::inferenceStarted, [this]() {
            m_generating = true;
            enableControls(false);
        });
    }
    
    // Ctrl+Enter to send
    m_generateShortcut = new QShortcut(QKeySequence(Qt::CTRL | Qt::Key_Return), this);
    connect(m_generateShortcut, &QShortcut::activated, this, &TextGenerationWidget::onSendMessage);
}

void TextGenerationWidget::setupUI()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setSpacing(0);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    
    setupHeaderCards();
    setupChatArea();
    setupMessageInput();
    setupSettingsPanel();
    
    mainLayout->addWidget(m_headerFrame);
    mainLayout->addWidget(m_chatScrollArea, 1);
    mainLayout->addWidget(m_inputFrame);
    mainLayout->addWidget(m_settingsPanel);
    
    m_settingsPanel->setVisible(false);
    enableControls(false);
    updateModelInfo();
}

static QFrame* createCard(QWidget *parent, const QString &title, const QStringList &items, const QString &bg)
{
    QFrame *card = new QFrame(parent);
    card->setFrameShape(QFrame::StyledPanel);
    card->setStyleSheet(QString("QFrame { background:%1; border-radius: 10px; padding: 12px; } QFrame QLabel.title { font-weight: 700; font-size: 16px; }").arg(bg));
    QVBoxLayout *v = new QVBoxLayout(card);
    QLabel *t = new QLabel(title, card);
    t->setObjectName("title");
    v->addWidget(t);
    for (const QString &s : items) {
        QLabel *l = new QLabel(QString("• %1").arg(s), card);
        l->setWordWrap(true);
        v->addWidget(l);
    }
    return card;
}

void TextGenerationWidget::setupHeaderCards()
{
    m_headerFrame = new QFrame(this);
    m_headerFrame->setMaximumHeight(200);
    QHBoxLayout *row = new QHBoxLayout(m_headerFrame);
    row->setSpacing(12);

    m_examplesCard = new QFrame(m_headerFrame);
    m_examplesCard->setFrameShape(QFrame::StyledPanel);
    m_examplesCard->setStyleSheet("QFrame { background:#FFE94D; border-radius:10px; padding:12px; } QLabel.title{font-weight:700;font-size:16px;}");
    QVBoxLayout *exV = new QVBoxLayout(m_examplesCard);
    QLabel *exT = new QLabel("Examples", m_examplesCard);
    exT->setObjectName("title");
    exV->addWidget(exT);

    const QStringList examples = {
        "What would happen if all of the world's clocks suddenly stopped?",
        "Create a plot twist for a classic fairy tale that will blow people's minds.",
        "Make a bot based on React or Laravel that writes incredibly bad poetry.",
        "Find hotels for a New Year trip to San Francisco; also check availability."
    };
    for (const QString &ex : examples) {
        QPushButton *b = new QPushButton(ex, m_examplesCard);
        b->setCursor(Qt::PointingHandCursor);
        b->setStyleSheet("QPushButton { text-align:left; padding:6px; border-radius:6px; background:white; } QPushButton:hover { background:#f2f2f2; }");
        connect(b, &QPushButton::clicked, this, &TextGenerationWidget::onExampleButtonClicked);
        m_exampleButtons.append(b);
        exV->addWidget(b);
    }

    QStringList caps = {"Understands prompts and generates coherent text","Supports adjustable sampling: temperature, top-p, top-k","Beam search and cache options"};
    QStringList lims = {"May produce incorrect information","Sensitive to prompt phrasing","Generation length and speed depend on model"};

    m_capabilitiesCard = createCard(this, "Capabilities", caps, "#7CF0B5");
    m_limitationsCard = createCard(this, "Limitations", lims, "#FF80CF");

    row->addWidget(m_examplesCard, 2);
    row->addWidget(m_capabilitiesCard, 1);
    row->addWidget(m_limitationsCard, 1);
}

void TextGenerationWidget::setupChatArea()
{
    m_chatScrollArea = new QScrollArea(this);
    m_chatScrollArea->setWidgetResizable(true);
    m_chatScrollArea->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_chatScrollArea->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_chatScrollArea->setStyleSheet("QScrollArea { border: none; background-color: white; }");
    
    m_chatContainer = new QWidget(m_chatScrollArea);
    m_chatLayout = new QVBoxLayout(m_chatContainer);
    m_chatLayout->setSpacing(8);
    m_chatLayout->setContentsMargins(16, 16, 16, 16);
    m_chatLayout->addStretch();
    
    m_chatScrollArea->setWidget(m_chatContainer);
}

void TextGenerationWidget::setupMessageInput()
{
    m_inputFrame = new QFrame(this);
    m_inputFrame->setStyleSheet("QFrame { background-color: white; border-top: 1px solid #E0E0E0; }");
    m_inputFrame->setMaximumHeight(120);
    
    m_inputLayout = new QHBoxLayout(m_inputFrame);
    m_inputLayout->setContentsMargins(16, 12, 16, 12);
    m_inputLayout->setSpacing(8);
    
    // Voice button
    m_voiceButton = new QPushButton("🎤", this);
    m_voiceButton->setFixedSize(32, 32);
    m_voiceButton->setStyleSheet("QPushButton { border: none; background-color: #F0F0F0; border-radius: 16px; font-size: 16px; } QPushButton:hover { background-color: #E0E0E0; }");
    connect(m_voiceButton, &QPushButton::clicked, this, &TextGenerationWidget::onStartVoiceRecording);
    m_inputLayout->addWidget(m_voiceButton);
    
    // Upload button
    m_uploadButton = new QPushButton("📎", this);
    m_uploadButton->setFixedSize(32, 32);
    m_uploadButton->setStyleSheet("QPushButton { border: none; background-color: #F0F0F0; border-radius: 16px; font-size: 16px; } QPushButton:hover { background-color: #E0E0E0; }");
    m_inputLayout->addWidget(m_uploadButton);
    
    // Link button
    m_linkButton = new QPushButton("🔗", this);
    m_linkButton->setFixedSize(32, 32);
    m_linkButton->setStyleSheet("QPushButton { border: none; background-color: #F0F0F0; border-radius: 16px; font-size: 16px; } QPushButton:hover { background-color: #E0E0E0; }");
    m_inputLayout->addWidget(m_linkButton);
    
    // Message input
    m_messageInput = new QTextEdit(this);
    m_messageInput->setMaximumHeight(80);
    m_messageInput->setPlaceholderText("Type your message here... (Ctrl+Enter to send)");
    m_messageInput->setStyleSheet("QTextEdit { border: 1px solid #E0E0E0; border-radius: 20px; padding: 8px; font-size: 14px; }");
    m_inputLayout->addWidget(m_messageInput, 1);
    
    // Send button
    m_sendButton = new QPushButton("➤", this);
    m_sendButton->setFixedSize(40, 40);
    m_sendButton->setStyleSheet("QPushButton { border: none; background-color: #007AFF; color: white; border-radius: 20px; font-size: 18px; font-weight: bold; } QPushButton:hover { background-color: #0056CC; } QPushButton:disabled { background-color: #CCCCCC; }");
    connect(m_sendButton, &QPushButton::clicked, this, &TextGenerationWidget::onSendMessage);
    m_inputLayout->addWidget(m_sendButton);
}

void TextGenerationWidget::setupSettingsPanel()
{
    m_settingsPanel = new QFrame(this);
    m_settingsPanel->setStyleSheet("QFrame { background-color: #F8F9FA; border-top: 1px solid #E0E0E0; }");
    m_settingsPanel->setMaximumHeight(300);
    
    QVBoxLayout *settingsLayout = new QVBoxLayout(m_settingsPanel);
    
    // Settings button
    m_settingsButton = new QPushButton("⚙️ Settings", this);
    m_settingsButton->setStyleSheet("QPushButton { border: none; background-color: transparent; font-size: 14px; color: #666; } QPushButton:hover { color: #333; }");
    connect(m_settingsButton, &QPushButton::clicked, this, &TextGenerationWidget::onSettingsButtonClicked);
    settingsLayout->addWidget(m_settingsButton);
    
    // Model section
    setupModelSection();
    settingsLayout->addWidget(m_modelGroup);
    
    // Generation settings
    setupGenerationSettings();
    settingsLayout->addWidget(m_settingsGroup);
}

void TextGenerationWidget::setupModelSection()
{
    m_modelGroup = new QGroupBox("Model", this);
    m_modelGroup->setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #E0E0E0; border-radius: 8px; margin-top: 8px; padding-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px 0 4px; }");
    QVBoxLayout *modelLayout = new QVBoxLayout(m_modelGroup);
    
    QHBoxLayout *pathLayout = new QHBoxLayout();
    m_modelPathEdit = new QLineEdit(this);
    m_modelPathEdit->setPlaceholderText("Select a model file (.onnx, .pt, .pb, .tflite, etc.)");
    m_modelPathEdit->setReadOnly(true);
    m_modelPathEdit->setStyleSheet("QLineEdit { border: 1px solid #E0E0E0; border-radius: 4px; padding: 4px; }");
    pathLayout->addWidget(m_modelPathEdit);
    
    m_loadModelButton = new QPushButton("Browse...", this);
    m_loadModelButton->setStyleSheet("QPushButton { border: 1px solid #007AFF; background-color: #007AFF; color: white; border-radius: 4px; padding: 4px 8px; } QPushButton:hover { background-color: #0056CC; }");
    connect(m_loadModelButton, &QPushButton::clicked, this, &TextGenerationWidget::onLoadModel);
    pathLayout->addWidget(m_loadModelButton);
    
    modelLayout->addLayout(pathLayout);
    
    m_modelStatusLabel = new QLabel("No model loaded", this);
    m_modelStatusLabel->setStyleSheet("color: red; font-weight: bold; font-size: 12px;");
    modelLayout->addWidget(m_modelStatusLabel);
    
    m_modelInfoLabel = new QLabel("", this);
    m_modelInfoLabel->setWordWrap(true);
    m_modelInfoLabel->setStyleSheet("font-size: 12px; color: #666;");
    modelLayout->addWidget(m_modelInfoLabel);
}

void TextGenerationWidget::setupGenerationSettings()
{
    m_settingsGroup = new QGroupBox("Generation Settings", this);
    m_settingsGroup->setStyleSheet("QGroupBox { font-weight: bold; border: 1px solid #E0E0E0; border-radius: 8px; margin-top: 8px; padding-top: 8px; } QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px 0 4px; }");
    QGridLayout *settingsLayout = new QGridLayout(m_settingsGroup);
    
    int row = 0;
    settingsLayout->addWidget(new QLabel("Max Length:"), row, 0);
    m_maxLengthSlider = new QSlider(Qt::Horizontal, this);
    m_maxLengthSlider->setRange(10, 2048);
    m_maxLengthSlider->setValue(100);
    settingsLayout->addWidget(m_maxLengthSlider, row, 1);
    
    m_maxLengthSpinBox = new QSpinBox(this);
    m_maxLengthSpinBox->setRange(10, 2048);
    m_maxLengthSpinBox->setValue(100);
    settingsLayout->addWidget(m_maxLengthSpinBox, row, 2);
    
    row++;
    settingsLayout->addWidget(new QLabel("Temperature:"), row, 0);
    m_temperatureSlider = new QSlider(Qt::Horizontal, this);
    m_temperatureSlider->setRange(0, 200);
    m_temperatureSlider->setValue(70);
    settingsLayout->addWidget(m_temperatureSlider, row, 1);
    
    m_temperatureSpinBox = new QDoubleSpinBox(this);
    m_temperatureSpinBox->setRange(0.0, 2.0);
    m_temperatureSpinBox->setSingleStep(0.1);
    m_temperatureSpinBox->setValue(0.7);
    settingsLayout->addWidget(m_temperatureSpinBox, row, 2);
    
    row++;
    settingsLayout->addWidget(new QLabel("Top P:"), row, 0);
    m_topPSlider = new QSlider(Qt::Horizontal, this);
    m_topPSlider->setRange(0, 100);
    m_topPSlider->setValue(90);
    settingsLayout->addWidget(m_topPSlider, row, 1);
    
    m_topPSpinBox = new QDoubleSpinBox(this);
    m_topPSpinBox->setRange(0.0, 1.0);
    m_topPSpinBox->setSingleStep(0.01);
    m_topPSpinBox->setValue(0.9);
    settingsLayout->addWidget(m_topPSpinBox, row, 2);
    
    row++;
    settingsLayout->addWidget(new QLabel("Top K:"), row, 0);
    m_topKSlider = new QSlider(Qt::Horizontal, this);
    m_topKSlider->setRange(1, 100);
    m_topKSlider->setValue(50);
    settingsLayout->addWidget(m_topKSlider, row, 1);
    
    m_topKSpinBox = new QSpinBox(this);
    m_topKSpinBox->setRange(1, 100);
    m_topKSpinBox->setValue(50);
    settingsLayout->addWidget(m_topKSpinBox, row, 2);
    
    row++;
    settingsLayout->addWidget(new QLabel("Repetition Penalty:"), row, 0);
    m_repetitionPenaltySlider = new QSlider(Qt::Horizontal, this);
    m_repetitionPenaltySlider->setRange(100, 200);
    m_repetitionPenaltySlider->setValue(120);
    settingsLayout->addWidget(m_repetitionPenaltySlider, row, 1);
    
    m_repetitionPenaltySpinBox = new QDoubleSpinBox(this);
    m_repetitionPenaltySpinBox->setRange(1.0, 2.0);
    m_repetitionPenaltySpinBox->setSingleStep(0.01);
    m_repetitionPenaltySpinBox->setValue(1.2);
    settingsLayout->addWidget(m_repetitionPenaltySpinBox, row, 2);
    
    row++;
    m_useBeamSearchCheck = new QCheckBox("Use Beam Search", this);
    settingsLayout->addWidget(m_useBeamSearchCheck, row, 0);
    
    settingsLayout->addWidget(new QLabel("Beam Width:"), row, 1);
    m_beamWidthSpinBox = new QSpinBox(this);
    m_beamWidthSpinBox->setRange(1, 10);
    m_beamWidthSpinBox->setValue(5);
    settingsLayout->addWidget(m_beamWidthSpinBox, row, 2);
    
    row++;
    m_doSampleCheck = new QCheckBox("Do Sampling", this);
    m_doSampleCheck->setChecked(true);
    settingsLayout->addWidget(m_doSampleCheck, row, 0);
    
    m_useCacheCheck = new QCheckBox("Use KV Cache", this);
    m_useCacheCheck->setChecked(true);
    settingsLayout->addWidget(m_useCacheCheck, row, 1);

    connect(m_maxLengthSlider, &QSlider::valueChanged, this, &TextGenerationWidget::onMaxLengthChanged);
    connect(m_maxLengthSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &TextGenerationWidget::onMaxLengthChanged);
    connect(m_temperatureSlider, &QSlider::valueChanged, this, &TextGenerationWidget::onTemperatureChanged);
    connect(m_temperatureSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &TextGenerationWidget::onTemperatureChanged);
    connect(m_topPSlider, &QSlider::valueChanged, this, &TextGenerationWidget::onTopPChanged);
    connect(m_topPSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &TextGenerationWidget::onTopPChanged);
    connect(m_topKSlider, &QSlider::valueChanged, this, &TextGenerationWidget::onTopKChanged);
    connect(m_topKSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &TextGenerationWidget::onTopKChanged);
    connect(m_repetitionPenaltySlider, &QSlider::valueChanged, this, &TextGenerationWidget::onRepetitionPenaltyChanged);
    connect(m_repetitionPenaltySpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &TextGenerationWidget::onRepetitionPenaltyChanged);
}

void TextGenerationWidget::setupVoiceSection()
{
    m_voiceGroup = new QGroupBox("Voice Input", this);
    QVBoxLayout *voiceLayout = new QVBoxLayout(m_voiceGroup);
    QHBoxLayout *recordingLayout = new QHBoxLayout();
    m_startRecordingButton = new QPushButton("🎤 Start Recording", this);
    m_startRecordingButton->setStyleSheet("QPushButton { background-color: #4CAF50; color: white; border: none; padding: 8px; border-radius: 4px; }");
    connect(m_startRecordingButton, &QPushButton::clicked, this, &TextGenerationWidget::onStartVoiceRecording);
    recordingLayout->addWidget(m_startRecordingButton);
    m_stopRecordingButton = new QPushButton("⏹️ Stop Recording", this);
    m_stopRecordingButton->setStyleSheet("QPushButton { background-color: #f44336; color: white; border: none; padding: 8px; border-radius: 4px; }");
    m_stopRecordingButton->setEnabled(false);
    connect(m_stopRecordingButton, &QPushButton::clicked, this, &TextGenerationWidget::onStopVoiceRecording);
    recordingLayout->addWidget(m_stopRecordingButton);
    m_playRecordingButton = new QPushButton("▶️ Play", this);
    m_playRecordingButton->setEnabled(false);
    connect(m_playRecordingButton, &QPushButton::clicked, this, &TextGenerationWidget::onPlayVoiceRecording);
    recordingLayout->addWidget(m_playRecordingButton);
    m_clearRecordingButton = new QPushButton("🗑️ Clear", this);
    m_clearRecordingButton->setEnabled(false);
    connect(m_clearRecordingButton, &QPushButton::clicked, this, &TextGenerationWidget::onClearVoiceRecording);
    recordingLayout->addWidget(m_clearRecordingButton);
    voiceLayout->addLayout(recordingLayout);
    m_voiceToTextButton = new QPushButton("🎵 Voice to Text", this);
    m_voiceToTextButton->setStyleSheet("QPushButton { background-color: #2196F3; color: white; border: none; padding: 8px; border-radius: 4px; }");
    m_voiceToTextButton->setEnabled(false);
    connect(m_voiceToTextButton, &QPushButton::clicked, this, &TextGenerationWidget::onVoiceToText);
    voiceLayout->addWidget(m_voiceToTextButton);
    m_recordingStatusLabel = new QLabel("Ready to record", this);
    voiceLayout->addWidget(m_recordingStatusLabel);
    m_recordingLevelBar = new QProgressBar(this);
    m_recordingLevelBar->setVisible(false);
    voiceLayout->addWidget(m_recordingLevelBar);
}

void TextGenerationWidget::setupOutputSection()
{
    m_outputGroup = new QGroupBox("Generated Text", this);
    QVBoxLayout *outputLayout = new QVBoxLayout(m_outputGroup);
    m_outputLabel = new QLabel("Generated output:", this);
    outputLayout->addWidget(m_outputLabel);
    m_outputTextEdit = new QTextEdit(this);
    m_outputTextEdit->setReadOnly(true);
    m_outputTextEdit->setPlaceholderText("Generated text will appear here...");
    outputLayout->addWidget(m_outputTextEdit);
    m_generationProgressBar = new QProgressBar(this);
    m_generationProgressBar->setVisible(false);
    outputLayout->addWidget(m_generationProgressBar);
}

void TextGenerationWidget::setupActions()
{
    m_actionsLayout = new QHBoxLayout();
    m_generateButton = new QPushButton("🚀 Generate Text", this);
    m_generateButton->setStyleSheet("QPushButton { background-color: #2196F3; color: white; border: none; padding: 10px; border-radius: 4px; font-weight: bold; }");
    connect(m_generateButton, &QPushButton::clicked, this, &TextGenerationWidget::onSendMessage);
    m_actionsLayout->addWidget(m_generateButton);
    m_clearButton = new QPushButton("🗑️ Clear All", this);
    connect(m_clearButton, &QPushButton::clicked, this, &TextGenerationWidget::onClear);
    m_actionsLayout->addWidget(m_clearButton);
    m_actionsLayout->addStretch();
}

void TextGenerationWidget::addMessage(const QString &text, bool isUser)
{
    ChatMessage msg;
    msg.text = text;
    msg.isUser = isUser;
    msg.timestamp = QDateTime::currentDateTime().toString("HH:mm");
    m_chatHistory.append(msg);
    
    // Remove the stretch item
    QLayoutItem *stretchItem = m_chatLayout->takeAt(m_chatLayout->count() - 1);
    delete stretchItem;
    
    // Add the message bubble
    MessageBubble *bubble = new MessageBubble(text, isUser, m_chatContainer);
    m_chatLayout->addWidget(bubble);
    
    // Add stretch back
    m_chatLayout->addStretch();
    
    scrollToBottom();
}

void TextGenerationWidget::scrollToBottom()
{
    QTimer::singleShot(100, [this]() {
        QScrollBar *scrollBar = m_chatScrollArea->verticalScrollBar();
        scrollBar->setValue(scrollBar->maximum());
    });
}

void TextGenerationWidget::onSendMessage()
{
    if (!m_modelManager || m_generating || !m_modelLoaded) return;
    
    QString message = m_messageInput->toPlainText().trimmed();
    if (message.isEmpty()) {
        return;
    }
    
    // Add user message to chat
    addMessage(message, true);
    
    // Clear input
    m_messageInput->clear();
    
    // Run inference
    int maxLength = m_maxLengthSpinBox->value();
    m_modelManager->runTextInference(message, maxLength);
}

void TextGenerationWidget::onLoadModel()
{
    QString modelPath = QFileDialog::getOpenFileName(
        this,
        "Select Model File",
        QStandardPaths::writableLocation(QStandardPaths::DocumentsLocation),
        "Model Files (*.onnx *.pt *.pth *.pb *.tflite *.bin *.safetensors);;All Files (*.*)"
    );
    if (!modelPath.isEmpty()) {
        m_modelPathEdit->setText(modelPath);
        m_modelStatusLabel->setText("Loading model...");
        m_modelStatusLabel->setStyleSheet("color: orange; font-weight: bold; font-size: 12px;");
        if (m_modelManager) {
            m_modelManager->loadModel(modelPath);
        }
    }
}

void TextGenerationWidget::onModelLoaded(const QString &modelPath)
{
    m_modelLoaded = true;
    m_currentModelPath = modelPath;
    m_modelStatusLabel->setText("✓ Model loaded successfully");
    m_modelStatusLabel->setStyleSheet("color: green; font-weight: bold; font-size: 12px;");
    updateModelInfo();
    enableControls(true);
}

void TextGenerationWidget::onModelLoadFailed(const QString &error)
{
    m_modelLoaded = false;
    m_modelStatusLabel->setText("✗ Failed to load model: " + error);
    m_modelStatusLabel->setStyleSheet("color: red; font-weight: bold; font-size: 12px;");
    enableControls(false);
}

void TextGenerationWidget::onInferenceCompleted(const InferenceResult &result)
{
    m_generating = false;
    enableControls(true);
    
    if (result.success) {
        addMessage(result.outputText, false);
    } else {
        addMessage("Error: " + result.errorMessage, false);
    }
}

void TextGenerationWidget::onClear()
{
    m_messageInput->clear();
    
    // Clear chat history
    QLayoutItem *item;
    while ((item = m_chatLayout->takeAt(0)) != nullptr) {
        if (item->widget()) {
            item->widget()->deleteLater();
        }
        delete item;
    }
    m_chatHistory.clear();
    m_chatLayout->addStretch();
    
    onClearVoiceRecording();
}

void TextGenerationWidget::onStartVoiceRecording()
{
    if (m_isRecording) return;
    QMessageBox::information(this, "Voice Recording", 
        "Voice recording functionality will be implemented in a future version.\nThis would use Qt6 Multimedia or a third-party audio library.");
    m_isRecording = true;
    m_startRecordingButton->setEnabled(false);
    m_stopRecordingButton->setEnabled(true);
    m_recordingStatusLabel->setText("🔴 Recording (Simulated)...");
    m_recordingLevelBar->setVisible(true);
    QTimer *levelTimer = new QTimer(this);
    connect(levelTimer, &QTimer::timeout, [this]() {
        if (m_isRecording) {
            int level = QRandomGenerator::global()->bounded(20, 100);
            m_recordingLevelBar->setValue(level);
        }
    });
    levelTimer->start(100);
}

void TextGenerationWidget::onStopVoiceRecording()
{
    if (!m_isRecording) return;
    m_isRecording = false;
    m_startRecordingButton->setEnabled(true);
    m_stopRecordingButton->setEnabled(false);
    m_playRecordingButton->setEnabled(true);
    m_clearRecordingButton->setEnabled(true);
    m_voiceToTextButton->setEnabled(true);
    m_recordingStatusLabel->setText("✓ Recording saved (Simulated)");
    m_recordingLevelBar->setVisible(false);
}

void TextGenerationWidget::onPlayVoiceRecording()
{
    QMessageBox::information(this, "Play Recording", "Voice playback functionality will be implemented in a future version.");
}

void TextGenerationWidget::onClearVoiceRecording()
{
    m_playRecordingButton->setEnabled(false);
    m_clearRecordingButton->setEnabled(false);
    m_voiceToTextButton->setEnabled(false);
    m_recordingStatusLabel->setText("Ready to record");
}

void TextGenerationWidget::onVoiceToText()
{
    QMessageBox::information(this, "Voice to Text", "Voice-to-text conversion would be implemented here.");
    m_messageInput->setPlainText("[Voice input converted to text would appear here]");
}

void TextGenerationWidget::onExampleButtonClicked()
{
    QPushButton *btn = qobject_cast<QPushButton*>(sender());
    if (!btn) return;
    m_messageInput->setPlainText(btn->text());
    m_messageInput->setFocus();
}

void TextGenerationWidget::onSettingsButtonClicked()
{
    m_settingsVisible = !m_settingsVisible;
    m_settingsPanel->setVisible(m_settingsVisible);
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
        m_temperatureSpinBox->setValue(value / 100.0);
    } else if (sender() == m_temperatureSpinBox) {
        m_temperatureSlider->setValue(static_cast<int>(value * 100));
    }
}

void TextGenerationWidget::onTopPChanged(double value)
{
    if (sender() == m_topPSlider) {
        m_topPSpinBox->setValue(value / 100.0);
    } else if (sender() == m_topPSpinBox) {
        m_topPSlider->setValue(static_cast<int>(value * 100));
    }
}

void TextGenerationWidget::onTopKChanged(int value)
{
    if (sender() == m_topKSlider) {
        m_topKSpinBox->setValue(value);
    } else if (sender() == m_topKSpinBox) {
        m_topKSlider->setValue(value);
    }
}

void TextGenerationWidget::onRepetitionPenaltyChanged(double value)
{
    if (sender() == m_repetitionPenaltySlider) {
        m_repetitionPenaltySpinBox->setValue(value / 100.0);
    } else if (sender() == m_repetitionPenaltySpinBox) {
        m_repetitionPenaltySlider->setValue(static_cast<int>(value * 100));
    }
}

void TextGenerationWidget::enableControls(bool enable)
{
    m_sendButton->setEnabled(enable && m_modelLoaded);
    m_messageInput->setEnabled(enable);
    m_voiceButton->setEnabled(enable && !m_isRecording);
}

void TextGenerationWidget::updateModelInfo()
{
    if (m_modelLoaded && !m_currentModelPath.isEmpty()) {
        QFileInfo fileInfo(m_currentModelPath);
        QString info = QString("Model: %1\nSize: %2 bytes\nFormat: %3")
                      .arg(fileInfo.fileName())
                      .arg(fileInfo.size())
                      .arg(fileInfo.suffix().toUpper());
        m_modelInfoLabel->setText(info);
    } else {
        m_modelInfoLabel->setText("");
    }
} 