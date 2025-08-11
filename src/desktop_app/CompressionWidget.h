#ifndef COMPRESSIONWIDGET_H
#define COMPRESSIONWIDGET_H

#include <QWidget>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QSlider>
#include <QSpinBox>
#include <QDoubleSpinBox>
#include <QCheckBox>
#include <QComboBox>
#include <QProgressBar>
#include <QTextEdit>
#include <QFileDialog>
#include <QMessageBox>

#include "ModelManager.h"

class CompressionWidget : public QWidget
{
    Q_OBJECT

public:
    explicit CompressionWidget(ModelManager *modelManager, QWidget *parent = nullptr);

private slots:
    void onBrowseModel();
    void onBrowseOutput();
    void onCompress();
    void onCompressionStarted();
    void onCompressionProgress(int percentage);
    void onCompressionCompleted(const CompressionResult &result);
    void onModelLoaded(const QString &modelPath);
    void onSparsityChanged(double value);
    void onCompressionLevelChanged(int value);
    void onClear();

private:
    void setupUI();
    void updateCompressionSettings();
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
    
    // UI Components - Output
    QGroupBox *m_outputGroup;
    QLineEdit *m_outputPathEdit;
    QPushButton *m_browseOutputButton;
    
    // UI Components - Compression Settings
    QGroupBox *m_settingsGroup;
    QSlider *m_sparsitySlider;
    QDoubleSpinBox *m_sparsitySpinBox;
    QLabel *m_sparsityLabel;
    
    QSlider *m_compressionLevelSlider;
    QSpinBox *m_compressionLevelSpinBox;
    QLabel *m_compressionLevelLabel;
    
    QCheckBox *m_useQuantizationCheck;
    QSpinBox *m_quantizationBitsSpinBox;
    QLabel *m_quantizationBitsLabel;
    
    QComboBox *m_strategyCombo;
    QLabel *m_strategyLabel;
    
    // UI Components - Progress
    QGroupBox *m_progressGroup;
    QProgressBar *m_progressBar;
    QLabel *m_statusLabel;
    
    // UI Components - Results
    QGroupBox *m_resultsGroup;
    QTextEdit *m_resultsText;
    
    // UI Components - Actions
    QPushButton *m_compressButton;
    QPushButton *m_clearButton;
    
    // State
    bool m_compressing;
    QString m_currentModelPath;
};

#endif // COMPRESSIONWIDGET_H 