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

#include "ModelManager.h"

class TextGenerationWidget : public QWidget
{
    Q_OBJECT

public:
    explicit TextGenerationWidget(ModelManager *modelManager, QWidget *parent = nullptr);

private slots:
    void onGenerate();
    void onClear();
    void onMaxLengthChanged(int value);
    void onTemperatureChanged(double value);
    void onTopPChanged(double value);

private:
    void setupUI();
    void enableControls(bool enable);
    
    // Model Manager
    ModelManager *m_modelManager;
    
    // UI Components - Input
    QGroupBox *m_inputGroup;
    QTextEdit *m_inputTextEdit;
    QLabel *m_inputLabel;
    
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
    
    QCheckBox *m_useBeamSearchCheck;
    QSpinBox *m_beamWidthSpinBox;
    QLabel *m_beamWidthLabel;
    
    // UI Components - Output
    QGroupBox *m_outputGroup;
    QTextEdit *m_outputTextEdit;
    QLabel *m_outputLabel;
    
    // UI Components - Actions
    QPushButton *m_generateButton;
    QPushButton *m_clearButton;
    
    // State
    bool m_generating;
};

#endif // TEXTGENERATIONWIDGET_H 