#include "SDRWindow.hpp"
#include <QVBoxLayout>
#include <QPushButton>

SDRWindow::SDRWindow() : sdr() { // Use default constructor for sdr
    setWindowTitle("SDR Text Encoder/Decoder");
    setupUI();
}

void SDRWindow::setupUI() {
    QWidget* centralWidget = new QWidget(this);
    QVBoxLayout* layout = new QVBoxLayout(centralWidget);

    // Input section
    layout->addWidget(new QLabel("Input Text:"));
    inputText = new QTextEdit();
    inputText->setPlaceholderText("Enter text to encode...");
    layout->addWidget(inputText);

    // Encode/Decode buttons
    QPushButton* encodeButton = new QPushButton("Encode");
    connect(encodeButton, &QPushButton::clicked, this, &SDRWindow::encodeText);
    layout->addWidget(encodeButton);

    // Output section
    layout->addWidget(new QLabel("Output:"));
    outputText = new QTextEdit();
    outputText->setReadOnly(true);
    layout->addWidget(outputText);

    // Stats section
    statsLabel = new QLabel();
    layout->addWidget(statsLabel);

    setCentralWidget(centralWidget);
    resize(600, 400);
}

void SDRWindow::encodeText() {
    QString input = inputText->toPlainText();
    auto encoded = sdr.encodeText(input.toStdString());
    std::string decoded = sdr.decode();

    // Prepare statistics
    QString stats = QString("Original size: %1 bytes\n").arg(input.size());
    stats += QString("Encoded size: %1 bytes\n").arg(encoded.getMemorySize());
    stats += QString("Compression ratio: %1:1\n").arg(
        static_cast<float>(input.size()) / encoded.getMemorySize(), 0, 'f', 2);
    
    // Show active positions
    QString activePos = "Active positions: ";
    for (const auto& pos : encoded.activePositions) {
        activePos += QString::number(pos) + " ";
    }

    // Update UI
    outputText->setText(QString::fromStdString(decoded) + "\n\n" + activePos);
    statsLabel->setText(stats);
}
