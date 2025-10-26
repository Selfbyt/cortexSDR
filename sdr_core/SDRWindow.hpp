#ifndef SDRWINDOW_HPP
#define SDRWINDOW_HPP

#include <QMainWindow>
#include <QTextEdit>
#include <QLabel>
#include "cortexSDR.hpp"

class SDRWindow : public QMainWindow {
    Q_OBJECT

public:
    SDRWindow();

private slots:
    void encodeText();

private:
    void setupUI();
    
    SparseDistributedRepresentation sdr;
    QTextEdit* inputText;
    QTextEdit* outputText;
    QLabel* statsLabel;
};

#endif // SDRWINDOW_HPP 