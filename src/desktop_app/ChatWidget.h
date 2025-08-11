#ifndef CHATWIDGET_H
#define CHATWIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QScrollArea>
#include <QTextEdit>
#include <QPushButton>
#include <QLabel>
#include <QListWidget>
#include <QListWidgetItem>
#include <QFrame>
#include <QSpacerItem>
#include <QTimer>
#include <QProgressBar>
#include <QGroupBox>
#include <QComboBox>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QSplitter>

#include "ModelManager.h"

struct ChatMessage {
    QString content;
    bool isUser;
    qint64 timestamp;
    double inferenceTimeMs;
    QString modelResponse;
};

class ChatBubble : public QFrame
{
    Q_OBJECT

public:
    explicit ChatBubble(const ChatMessage &message, QWidget *parent = nullptr);

private:
    void setupUI();
    ChatMessage m_message;
    QLabel *m_contentLabel;
    QLabel *m_timeLabel;
    QLabel *m_metricsLabel;
};

class ChatWidget : public QWidget
{
    Q_OBJECT

public:
    explicit ChatWidget(ModelManager *modelManager, QWidget *parent = nullptr);

private slots:
    void onSendMessage();
    void onClearChat();
    void onModelLoaded(const QString &modelPath);
    void onInferenceCompleted(const InferenceResult &result);
    void onModelLoadFailed(const QString &error);
    void onMaxLengthChanged(int value);
    void onTemperatureChanged(double value);
    void onTopPChanged(double value);
    void onEnterPressed();

private:
    void setupUI();
    void setupChatArea();
    void setupInputArea();
    void setupSettingsArea();
    void addMessage(const ChatMessage &message);
    void scrollToBottom();
    void enableControls(bool enable);
    void updateModelInfo();
    bool eventFilter(QObject *obj, QEvent *event) override;
    
    // Model Manager
    ModelManager *m_modelManager;
    
    // UI Components - Main Layout
    QSplitter *m_mainSplitter;
    QWidget *m_chatContainer;
    QWidget *m_settingsContainer;
    
    // UI Components - Chat Area
    QVBoxLayout *m_chatLayout;
    QScrollArea *m_scrollArea;
    QWidget *m_chatContent;
    QListWidget *m_messageList;
    
    // UI Components - Input Area
    QHBoxLayout *m_inputLayout;
    QTextEdit *m_inputEdit;
    QPushButton *m_sendButton;
    QPushButton *m_clearButton;
    
    // UI Components - Settings
    QGroupBox *m_settingsGroup;
    QGridLayout *m_settingsLayout;
    
    QLabel *m_modelInfoLabel;
    QLabel *m_modelStatusLabel;
    
    QLabel *m_maxLengthLabel;
    QSpinBox *m_maxLengthSpinBox;
    
    QLabel *m_temperatureLabel;
    QDoubleSpinBox *m_temperatureSpinBox;
    
    QLabel *m_topPLabel;
    QDoubleSpinBox *m_topPSpinBox;
    
    QCheckBox *m_streamingCheck;
    QCheckBox *m_showMetricsCheck;
    
    // UI Components - Progress
    QProgressBar *m_progressBar;
    QLabel *m_statusLabel;
    
    // State
    bool m_generating;
    QString m_currentModelPath;
    QList<ChatMessage> m_chatHistory;
    
    // Auto-scroll timer
    QTimer *m_scrollTimer;
};

#endif // CHATWIDGET_H 