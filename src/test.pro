QT       += core gui
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = sdr_test
TEMPLATE = app

SOURCES += \
    test.cpp \
    SDRWindow.cpp \
    cortexSDR.cpp \
    encoders/AudioEncoding.cpp \
    encoders/DateTimeEncoding.cpp \
    encoders/ImageEncoding.cpp \
    encoders/NumberEncoding.cpp \
    encoders/VideoEncoding.cpp \
    encoders/WordEncoding.cpp

HEADERS += \
    SDRWindow.hpp \
    encoders/AudioEncoding.hpp \
    encoders/DateTimeEncoding.hpp \
    encoders/ImageEncoding.hpp \
    encoders/NumberEncoding.hpp \
    encoders/VideoEncoding.hpp \
    encoders/WordEncoding.hpp \
    encoders/CharacterEncoding.hpp

INCLUDEPATH += src/

CONFIG += c++17

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target