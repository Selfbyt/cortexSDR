#ifndef AUDIOINFERENCEWIDGET_H
#define AUDIOINFERENCEWIDGET_H

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

#include "ModelManager.h"

class AudioInferenceWidget : public QWidget
{
    Q_OBJECT

public:
    explicit AudioInferenceWidget(ModelManager *modelManager, QWidget *parent = nullptr);

private slots:
    void onBrowseAudio();
    void onProcessAudio();
    void onClear();
    void onSampleRateChanged(int value);
    void onChannelsChanged(int value);

private:
    void setupUI();
    void enableControls(bool enable);
    void updateAudioInfo();
    
    // Model Manager
    ModelManager *m_modelManager;
    
    // UI Components - Audio Input
    QGroupBox *m_audioGroup;
    QLineEdit *m_audioPathEdit;
    QPushButton *m_browseAudioButton;
    QLabel *m_audioFormatLabel;
    QLabel *m_audioDurationLabel;
    QLabel *m_audioSizeLabel;
    
    // UI Components - Audio Settings
    QGroupBox *m_settingsGroup;
    QComboBox *m_sampleRateCombo;
    QLabel *m_sampleRateLabel;
    
    QComboBox *m_channelsCombo;
    QLabel *m_channelsLabel;
    
    QComboBox *m_audioFormatCombo;
    QLabel *m_audioFormatLabel2;
    
    QSlider *m_volumeSlider;
    QDoubleSpinBox *m_volumeSpinBox;
    QLabel *m_volumeLabel;
    
    QCheckBox *m_normalizeCheck;
    QCheckBox *m_removeNoiseCheck;
    
    // UI Components - Output
    QGroupBox *m_outputGroup;
    QTextEdit *m_outputTextEdit;
    QLabel *m_outputLabel;
    
    QLineEdit *m_outputAudioPathEdit;
    QPushButton *m_browseOutputAudioButton;
    QPushButton *m_playOutputButton;
    
    // UI Components - Actions
    QPushButton *m_processButton;
    QPushButton *m_clearButton;
    
    // State
    bool m_processing;
    QString m_currentAudioPath;
};

#endif // AUDIOINFERENCEWIDGET_H 