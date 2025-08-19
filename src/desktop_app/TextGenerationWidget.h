#ifndef TEXTGENERATIONWIDGET_H
#define TEXTGENERATIONWIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSpinBox>
#include <QComboBox>
#include <QTextEdit>
#include <QSlider>
#include <QCheckBox>
#include <QFileDialog>
#include <QProgressBar>
#include <QBuffer>
#include <QFrame>
#include <QVector>
#include <QShortcut>
#include <QScrollArea>
#include <QListWidget>
#include <QListWidgetItem>

#include "ModelManager.h"

struct ChatMessage {
    QString text;
    bool isUser;
    QString timestamp;
};

class MessageBubble : public QFrame
{
    Q_OBJECT
public:
    MessageBubble(const QString &text, bool isUser, QWidget *parent = nullptr);
    
private:
    void setupUI();
    QString m_text;
    bool m_isUser;
};

class TextGenerationWidget : public QWidget
{
    Q_OBJECT

public:
    explicit TextGenerationWidget(ModelManager *modelManager, QWidget *parent = nullptr);

private slots:
    void onSendMessage();
    void onClear();
    void onMaxLengthChanged(int value);
    void onTemperatureChanged(double value);
    void onTopPChanged(double value);
    void onTopKChanged(int value);
    void onRepetitionPenaltyChanged(double value);
    void onLoadModel();
    void onModelLoaded(const QString &modelPath);
    void onModelLoadFailed(const QString &error);
    void onInferenceCompleted(const InferenceResult &result);
    void onStartVoiceRecording();
    void onStopVoiceRecording();
    void onPlayVoiceRecording();
    void onClearVoiceRecording();
    void onVoiceToText();
    void onExampleButtonClicked();
    void onSettingsButtonClicked();

private:
    void setupUI();
    void setupHeaderCards();
    void setupChatArea();
    void setupMessageInput();
    void setupModelSection();
    void setupSettingsPanel();
    void setupVoiceSection();
    void setupGenerationSettings();
    void setupOutputSection();
    void setupActions();
    void enableControls(bool enable);
    void updateModelInfo();
    void saveVoiceRecording();
    void loadVoiceRecording();
    void addMessage(const QString &text, bool isUser);
    void scrollToBottom();
    
    // Model Manager
    ModelManager *m_modelManager;
    
    // Header (Examples/Capabilities/Limitations)
    QFrame *m_headerFrame;
    QFrame *m_examplesCard;
    QFrame *m_capabilitiesCard;
    QFrame *m_limitationsCard;
    QVector<QPushButton*> m_exampleButtons;
    
    // Chat Area
    QScrollArea *m_chatScrollArea;
    QWidget *m_chatContainer;
    QVBoxLayout *m_chatLayout;
    QList<ChatMessage> m_chatHistory;
    
    // Message Input
    QFrame *m_inputFrame;
    QHBoxLayout *m_inputLayout;
    QPushButton *m_voiceButton;
    QPushButton *m_uploadButton;
    QPushButton *m_linkButton;
    QTextEdit *m_messageInput;
    QPushButton *m_sendButton;
    
    // Settings Panel
    QFrame *m_settingsPanel;
    QPushButton *m_settingsButton;
    bool m_settingsVisible;
    
    // UI Components - Model Section
    QGroupBox *m_modelGroup;
    QLineEdit *m_modelPathEdit;
    QPushButton *m_loadModelButton;
    QLabel *m_modelStatusLabel;
    QLabel *m_modelInfoLabel;
    
    // UI Components - Voice Section
    QGroupBox *m_voiceGroup;
    QPushButton *m_startRecordingButton;
    QPushButton *m_stopRecordingButton;
    QPushButton *m_playRecordingButton;
    QPushButton *m_clearRecordingButton;
    QPushButton *m_voiceToTextButton;
    QLabel *m_recordingStatusLabel;
    QProgressBar *m_recordingLevelBar;
    
    // UI Components - Generation Settings
    QGroupBox *m_settingsGroup;
    QSlider *m_maxLengthSlider;
    QSpinBox *m_maxLengthSpinBox;
    QLabel *m_maxLengthLabel;
    
    QSlider *m_temperatureSlider;
    QDoubleSpinBox *m_temperatureSpinBox;
    QLabel *m_temperatureLabel;
    
    QSlider *m_topPSlider;
    QDoubleSpinBox *m_topPSpinBox;
    QLabel *m_topPLabel;
    
    QSlider *m_topKSlider;
    QSpinBox *m_topKSpinBox;
    QLabel *m_topKLabel;
    
    QSlider *m_repetitionPenaltySlider;
    QDoubleSpinBox *m_repetitionPenaltySpinBox;
    QLabel *m_repetitionPenaltyLabel;
    
    QCheckBox *m_useBeamSearchCheck;
    QSpinBox *m_beamWidthSpinBox;
    QLabel *m_beamWidthLabel;
    
    QCheckBox *m_doSampleCheck;
    QCheckBox *m_useCacheCheck;
    
    // UI Components - Output
    QGroupBox *m_outputGroup;
    QTextEdit *m_outputTextEdit;
    QLabel *m_outputLabel;
    QProgressBar *m_generationProgressBar;
    
    // UI Components - Actions
    QHBoxLayout *m_actionsLayout;
    QPushButton *m_generateButton;
    QPushButton *m_clearButton;
    QShortcut *m_generateShortcut;
    
    // Audio components (placeholder for future implementation)
    QString m_voiceRecordingPath;
    bool m_isRecording;
    
    // State
    bool m_generating;
    bool m_modelLoaded;
    QString m_currentModelPath;
};

#endif // TEXTGENERATIONWIDGET_H 