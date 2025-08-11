#include "ChatWidget.h"
#include <QDateTime>
#include <QFileInfo>
#include <QMessageBox>
#include <QApplication>
#include <QStyle>
#include <QKeyEvent>
#include <QScrollBar>
#include <QDoubleSpinBox>

ChatBubble::ChatBubble(const ChatMessage &message, QWidget *parent)
    : QFrame(parent)
    , m_message(message)
{
    setupUI();
}

void ChatBubble::setupUI()
{
    setFrameStyle(QFrame::Box);
    setLineWidth(1);
    
    QVBoxLayout *layout = new QVBoxLayout(this);
    layout->setContentsMargins(10, 8, 10, 8);
    
    // Content label
    m_contentLabel = new QLabel(m_message.content, this);
    m_contentLabel->setWordWrap(true);
    m_contentLabel->setTextInteractionFlags(Qt::TextSelectableByMouse);
    m_contentLabel->setStyleSheet("QLabel { padding: 8px; border-radius: 8px; }");
    
    // Time label
    m_timeLabel = new QLabel(QDateTime::fromMSecsSinceEpoch(m_message.timestamp).toString("hh:mm:ss"), this);
    m_timeLabel->setStyleSheet("QLabel { color: #666; font-size: 10px; }");
    
    // Metrics label (for AI responses)
    if (!m_message.isUser && m_message.inferenceTimeMs > 0) {
        m_metricsLabel = new QLabel(QString("Response time: %1 ms").arg(m_message.inferenceTimeMs, 0, 'f', 1), this);
        m_metricsLabel->setStyleSheet("QLabel { color: #666; font-size: 10px; }");
        layout->addWidget(m_metricsLabel);
    } else {
        m_metricsLabel = nullptr;
    }
    
    layout->addWidget(m_contentLabel);
    layout->addWidget(m_timeLabel);
    
    // Style based on message type
    if (m_message.isUser) {
        setStyleSheet("ChatBubble { background-color: #007AFF; border-radius: 12px; }");
        m_contentLabel->setStyleSheet("QLabel { background-color: #007AFF; color: white; padding: 8px; border-radius: 8px; }");
        layout->setAlignment(Qt::AlignRight);
    } else {
        setStyleSheet("ChatBubble { background-color: #F0F0F0; border-radius: 12px; }");
        m_contentLabel->setStyleSheet("QLabel { background-color: #F0F0F0; color: black; padding: 8px; border-radius: 8px; }");
        layout->setAlignment(Qt::AlignLeft);
    }
}

ChatWidget::ChatWidget(ModelManager *modelManager, QWidget *parent)
    : QWidget(parent)
    , m_modelManager(modelManager)
    , m_generating(false)
    , m_scrollTimer(new QTimer(this))
{
    setupUI();
    
    if (m_modelManager) {
        connect(m_modelManager, &ModelManager::modelLoaded, this, &ChatWidget::onModelLoaded);
        connect(m_modelManager, &ModelManager::modelLoadFailed, this, &ChatWidget::onModelLoadFailed);
        connect(m_modelManager, &ModelManager::inferenceCompleted, this, &ChatWidget::onInferenceCompleted);
    }
    
    // Setup auto-scroll timer
    connect(m_scrollTimer, &QTimer::timeout, this, &ChatWidget::scrollToBottom);
    m_scrollTimer->setSingleShot(true);
}

void ChatWidget::setupUI()
{
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->setContentsMargins(0, 0, 0, 0);
    
    // Create splitter for chat and settings
    m_mainSplitter = new QSplitter(Qt::Horizontal, this);
    
    // Chat container
    m_chatContainer = new QWidget(this);
    m_chatLayout = new QVBoxLayout(m_chatContainer);
    m_chatLayout->setContentsMargins(0, 0, 0, 0);
    
    setupChatArea();
    setupInputArea();
    
    // Settings container
    m_settingsContainer = new QWidget(this);
    setupSettingsArea();
    
    m_mainSplitter->addWidget(m_chatContainer);
    m_mainSplitter->addWidget(m_settingsContainer);
    m_mainSplitter->setSizes({800, 300});
    
    mainLayout->addWidget(m_mainSplitter);
    
    // Progress bar and status
    m_progressBar = new QProgressBar(this);
    m_progressBar->setVisible(false);
    mainLayout->addWidget(m_progressBar);
    
    m_statusLabel = new QLabel("Ready", this);
    m_statusLabel->setStyleSheet("QLabel { color: #666; padding: 4px; }");
    mainLayout->addWidget(m_statusLabel);
    
    // Initial state
    enableControls(false);
    qDebug() << "ChatWidget constructor - initial state, input edit enabled:" << m_inputEdit->isEnabled();
}

void ChatWidget::setupChatArea()
{
    // Chat header
    QLabel *headerLabel = new QLabel("Chat Interface", this);
    headerLabel->setStyleSheet("QLabel { font-size: 16px; font-weight: bold; padding: 8px; background-color: #f8f9fa; border-bottom: 1px solid #dee2e6; }");
    m_chatLayout->addWidget(headerLabel);
    
    // Message list
    m_messageList = new QListWidget(this);
    m_messageList->setStyleSheet("QListWidget { border: none; background-color: white; }");
    m_messageList->setVerticalScrollBarPolicy(Qt::ScrollBarAsNeeded);
    m_messageList->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    m_messageList->setSpacing(8);
    m_chatLayout->addWidget(m_messageList);
    
    // Add welcome message
    ChatMessage welcomeMsg;
    welcomeMsg.content = "Hello! I'm your AI assistant. Load a model to start chatting.";
    welcomeMsg.isUser = false;
    welcomeMsg.timestamp = QDateTime::currentMSecsSinceEpoch();
    welcomeMsg.inferenceTimeMs = 0;
    addMessage(welcomeMsg);
}

void ChatWidget::setupInputArea()
{
    QFrame *inputFrame = new QFrame(this);
    inputFrame->setStyleSheet("QFrame { background-color: #f8f9fa; border-top: 1px solid #dee2e6; }");
    QHBoxLayout *inputFrameLayout = new QHBoxLayout(inputFrame);
    
    // Input text edit
    m_inputEdit = new QTextEdit(this);
    m_inputEdit->setMaximumHeight(80);
    m_inputEdit->setPlaceholderText("Type your message here... (Press Enter to send, Shift+Enter for new line)");
    m_inputEdit->setStyleSheet("QTextEdit { border: 1px solid #dee2e6; border-radius: 8px; padding: 8px; }");
    
    // Send button
    m_sendButton = new QPushButton("Send", this);
    m_sendButton->setStyleSheet("QPushButton { background-color: #007AFF; color: white; border: none; padding: 8px 16px; border-radius: 6px; } QPushButton:hover { background-color: #0056CC; } QPushButton:disabled { background-color: #ccc; }");
    m_sendButton->setMaximumWidth(80);
    
    // Clear button
    m_clearButton = new QPushButton("Clear", this);
    m_clearButton->setStyleSheet("QPushButton { background-color: #6c757d; color: white; border: none; padding: 8px 16px; border-radius: 6px; } QPushButton:hover { background-color: #545b62; }");
    m_clearButton->setMaximumWidth(80);
    
    inputFrameLayout->addWidget(m_inputEdit);
    inputFrameLayout->addWidget(m_sendButton);
    inputFrameLayout->addWidget(m_clearButton);
    
    m_chatLayout->addWidget(inputFrame);
    
    // Connect signals
    connect(m_sendButton, &QPushButton::clicked, this, &ChatWidget::onSendMessage);
    connect(m_clearButton, &QPushButton::clicked, this, &ChatWidget::onClearChat);
    
    qDebug() << "ChatWidget signals connected - send button:" << m_sendButton << "clear button:" << m_clearButton;
    
    // Handle Enter key
    m_inputEdit->installEventFilter(this);
    
    // Ensure the text edit is always enabled for testing
    m_inputEdit->setEnabled(true);
    m_inputEdit->setFocusPolicy(Qt::StrongFocus);
}

void ChatWidget::setupSettingsArea()
{
    QVBoxLayout *settingsLayout = new QVBoxLayout(m_settingsContainer);
    
    // Settings header
    QLabel *settingsHeader = new QLabel("Settings", this);
    settingsHeader->setStyleSheet("QLabel { font-size: 14px; font-weight: bold; padding: 8px; background-color: #f8f9fa; border-bottom: 1px solid #dee2e6; }");
    settingsLayout->addWidget(settingsHeader);
    
    // Model info group
    m_settingsGroup = new QGroupBox("Model Settings", this);
    m_settingsLayout = new QGridLayout(m_settingsGroup);
    
    // Model info
    m_modelInfoLabel = new QLabel("No model loaded", this);
    m_modelInfoLabel->setStyleSheet("QLabel { color: #dc3545; font-weight: bold; }");
    m_settingsLayout->addWidget(new QLabel("Model:"), 0, 0);
    m_settingsLayout->addWidget(m_modelInfoLabel, 0, 1);
    
    m_modelStatusLabel = new QLabel("Not ready", this);
    m_modelStatusLabel->setStyleSheet("QLabel { color: #dc3545; }");
    m_settingsLayout->addWidget(new QLabel("Status:"), 1, 0);
    m_settingsLayout->addWidget(m_modelStatusLabel, 1, 1);
    
    // Generation settings
    m_settingsLayout->addWidget(new QLabel("Max Length:"), 2, 0);
    m_maxLengthSpinBox = new QSpinBox(this);
    m_maxLengthSpinBox->setRange(10, 2000);
    m_maxLengthSpinBox->setValue(100);
    m_settingsLayout->addWidget(m_maxLengthSpinBox, 2, 1);
    
    m_settingsLayout->addWidget(new QLabel("Temperature:"), 3, 0);
    m_temperatureSpinBox = new QDoubleSpinBox(this);
    m_temperatureSpinBox->setRange(0.0, 2.0);
    m_temperatureSpinBox->setSingleStep(0.1);
    m_temperatureSpinBox->setValue(0.7);
    m_settingsLayout->addWidget(m_temperatureSpinBox, 3, 1);
    
    m_settingsLayout->addWidget(new QLabel("Top P:"), 4, 0);
    m_topPSpinBox = new QDoubleSpinBox(this);
    m_topPSpinBox->setRange(0.0, 1.0);
    m_topPSpinBox->setSingleStep(0.01);
    m_topPSpinBox->setValue(0.9);
    m_settingsLayout->addWidget(m_topPSpinBox, 4, 1);
    
    // Options
    m_streamingCheck = new QCheckBox("Enable Streaming", this);
    m_streamingCheck->setChecked(false);
    m_settingsLayout->addWidget(m_streamingCheck, 5, 0, 1, 2);
    
    m_showMetricsCheck = new QCheckBox("Show Response Metrics", this);
    m_showMetricsCheck->setChecked(true);
    m_settingsLayout->addWidget(m_showMetricsCheck, 6, 0, 1, 2);
    
    settingsLayout->addWidget(m_settingsGroup);
    settingsLayout->addStretch();
    
    // Connect signals
    connect(m_maxLengthSpinBox, QOverload<int>::of(&QSpinBox::valueChanged), this, &ChatWidget::onMaxLengthChanged);
    connect(m_temperatureSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &ChatWidget::onTemperatureChanged);
    connect(m_topPSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged), this, &ChatWidget::onTopPChanged);
}

void ChatWidget::onSendMessage()
{
    qDebug() << "onSendMessage called - m_generating:" << m_generating << "m_modelManager:" << m_modelManager << "isModelLoaded:" << (m_modelManager ? m_modelManager->isModelLoaded() : false);
    
    if (m_generating || !m_modelManager || !m_modelManager->isModelLoaded()) {
        qDebug() << "Send blocked - generating:" << m_generating << "no model manager:" << !m_modelManager << "model not loaded:" << (m_modelManager ? !m_modelManager->isModelLoaded() : true);
        return;
    }
    
    QString message = m_inputEdit->toPlainText().trimmed();
    qDebug() << "Message text:" << message << "isEmpty:" << message.isEmpty();
    if (message.isEmpty()) {
        return;
    }
    
    // Add user message
    ChatMessage userMsg;
    userMsg.content = message;
    userMsg.isUser = true;
    userMsg.timestamp = QDateTime::currentMSecsSinceEpoch();
    userMsg.inferenceTimeMs = 0;
    addMessage(userMsg);
    
    // Clear input
    m_inputEdit->clear();
    
    // Start generation
    m_generating = true;
    enableControls(false);
    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, 0); // Indeterminate progress
    m_statusLabel->setText("Generating response...");
    
    // Run inference
    m_modelManager->runTextInference(message, m_maxLengthSpinBox->value());
}

void ChatWidget::onClearChat()
{
    m_messageList->clear();
    m_chatHistory.clear();
    
    // Add welcome message back
    ChatMessage welcomeMsg;
    welcomeMsg.content = "Chat cleared. Load a model to start chatting.";
    welcomeMsg.isUser = false;
    welcomeMsg.timestamp = QDateTime::currentMSecsSinceEpoch();
    welcomeMsg.inferenceTimeMs = 0;
    addMessage(welcomeMsg);
}

void ChatWidget::onModelLoaded(const QString &modelPath)
{
    m_currentModelPath = modelPath;
    m_generating = false; // Ensure generating flag is reset
    updateModelInfo();
    enableControls(true);
    m_statusLabel->setText("Model loaded successfully");
    
    // Set focus to input area and ensure it's enabled
    m_inputEdit->setFocus();
    m_inputEdit->setEnabled(true);
    qDebug() << "Model loaded, input edit enabled:" << m_inputEdit->isEnabled() << "has focus:" << m_inputEdit->hasFocus();
}

void ChatWidget::onModelLoadFailed(const QString &error)
{
    m_currentModelPath.clear();
    updateModelInfo();
    enableControls(false);
    m_statusLabel->setText("Model loading failed: " + error);
}

void ChatWidget::onInferenceCompleted(const InferenceResult &result)
{
    m_generating = false;
    m_progressBar->setVisible(false);
    enableControls(true);
    
    if (result.success) {
        // Add AI response
        ChatMessage aiMsg;
        aiMsg.content = result.outputText;
        aiMsg.isUser = false;
        aiMsg.timestamp = QDateTime::currentMSecsSinceEpoch();
        aiMsg.inferenceTimeMs = result.inferenceTimeMs;
        aiMsg.modelResponse = result.outputText;
        addMessage(aiMsg);
        
        m_statusLabel->setText(QString("Response generated in %1 ms").arg(result.inferenceTimeMs, 0, 'f', 1));
    } else {
        // Add error message
        ChatMessage errorMsg;
        errorMsg.content = "Sorry, I encountered an error: " + result.errorMessage;
        errorMsg.isUser = false;
        errorMsg.timestamp = QDateTime::currentMSecsSinceEpoch();
        errorMsg.inferenceTimeMs = 0;
        addMessage(errorMsg);
        
        m_statusLabel->setText("Inference failed: " + result.errorMessage);
    }
}

void ChatWidget::onMaxLengthChanged(int value)
{
    // Update any running inference if needed
}

void ChatWidget::onTemperatureChanged(double value)
{
    // Update any running inference if needed
}

void ChatWidget::onTopPChanged(double value)
{
    // Update any running inference if needed
}

void ChatWidget::onEnterPressed()
{
    // Handle Enter key press
    QKeyEvent *keyEvent = static_cast<QKeyEvent*>(QApplication::focusWidget()->property("lastKeyEvent").value<void*>());
    if (keyEvent && keyEvent->key() == Qt::Key_Return && !keyEvent->modifiers().testFlag(Qt::ShiftModifier)) {
        onSendMessage();
    }
}

void ChatWidget::addMessage(const ChatMessage &message)
{
    m_chatHistory.append(message);
    
    // Create chat bubble
    ChatBubble *bubble = new ChatBubble(message, this);
    QListWidgetItem *item = new QListWidgetItem(m_messageList);
    item->setSizeHint(bubble->sizeHint());
    m_messageList->setItemWidget(item, bubble);
    
    // Scroll to bottom
    m_scrollTimer->start(100);
}

void ChatWidget::scrollToBottom()
{
    if (m_messageList->count() > 0) {
        m_messageList->scrollToBottom();
    }
}

void ChatWidget::enableControls(bool enable)
{
    bool inputEnabled = enable && !m_generating;
    m_sendButton->setEnabled(inputEnabled);
    // Always enable the input edit for now to test
    m_inputEdit->setEnabled(true);
    m_maxLengthSpinBox->setEnabled(enable);
    m_temperatureSpinBox->setEnabled(enable);
    m_topPSpinBox->setEnabled(enable);
    m_streamingCheck->setEnabled(enable);
    m_showMetricsCheck->setEnabled(enable);
    
    // Debug output
    qDebug() << "enableControls called with enable=" << enable << "m_generating=" << m_generating << "inputEnabled=" << inputEnabled;
}

void ChatWidget::updateModelInfo()
{
    if (m_currentModelPath.isEmpty()) {
        m_modelInfoLabel->setText("No model loaded");
        m_modelInfoLabel->setStyleSheet("QLabel { color: #dc3545; font-weight: bold; }");
        m_modelStatusLabel->setText("Not ready");
        m_modelStatusLabel->setStyleSheet("QLabel { color: #dc3545; }");
    } else {
        QFileInfo fileInfo(m_currentModelPath);
        m_modelInfoLabel->setText(fileInfo.fileName());
        m_modelInfoLabel->setStyleSheet("QLabel { color: #28a745; font-weight: bold; }");
        m_modelStatusLabel->setText("Ready");
        m_modelStatusLabel->setStyleSheet("QLabel { color: #28a745; }");
    }
}

bool ChatWidget::eventFilter(QObject *obj, QEvent *event)
{
    if (obj == m_inputEdit && event->type() == QEvent::KeyPress) {
        QKeyEvent *keyEvent = static_cast<QKeyEvent*>(event);
        if (keyEvent->key() == Qt::Key_Return && !keyEvent->modifiers().testFlag(Qt::ShiftModifier)) {
            onSendMessage();
            return true;
        }
    }
    return QWidget::eventFilter(obj, event);
} 