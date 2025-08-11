#ifndef INFERENCEWIDGET_H
#define INFERENCEWIDGET_H

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
#include <QProgressBar>
#include <QTabWidget>
#include <QFileDialog>
#include <QSlider>
#include <QCheckBox>

#include "ModelManager.h"
#include "TextGenerationWidget.h"
#include "AudioInferenceWidget.h"

class InferenceWidget : public QWidget
{
    Q_OBJECT

public:
    explicit InferenceWidget(ModelManager *modelManager, QWidget *parent = nullptr);

private slots:
    void onModelLoaded(const QString &modelPath);
    void onInferenceCompleted(const InferenceResult &result);
    void onInferenceStarted();
    void onModelTypeChanged(int index);
    void onLoadCompressedModel();

private:
    void setupUI();
    void updateModelInfo();
    void enableControls(bool enable);
    
    // Model Manager
    ModelManager *m_modelManager;
    
    // UI Components - Model Selection
    QGroupBox *m_modelGroup;
    QLineEdit *m_modelPathEdit;
    QPushButton *m_browseModelButton;
    QLabel *m_modelFormatLabel;
    QLabel *m_modelSizeLabel;
    QComboBox *m_modelTypeCombo;
    
    // UI Components - Model Type Tabs
    QTabWidget *m_inferenceTabs;
    TextGenerationWidget *m_textWidget;
    AudioInferenceWidget *m_audioWidget;
    
    // UI Components - Progress
    QGroupBox *m_progressGroup;
    QProgressBar *m_progressBar;
    QLabel *m_statusLabel;
    
    // UI Components - Results
    QGroupBox *m_resultsGroup;
    QTextEdit *m_resultsText;
    
    // State
    bool m_inferencing;
    QString m_currentModelPath;
    QString m_currentModelType;
};

#endif // INFERENCEWIDGET_H 