#include <QApplication>
#include "SDRWindow.hpp"

int main(int argc, char *argv[]) {
    QApplication app(argc, argv);
    SDRWindow window;
    window.show();
    return app.exec();
}
