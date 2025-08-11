#ifndef PERFORMANCEMONITOR_H
#define PERFORMANCEMONITOR_H

#include <QObject>
#include <QTimer>
#include <QProcess>
#include <QFile>
#include <QTextStream>
#include <QString>
#include <QList>

struct PerformanceMetrics {
    double cpuUsage;
    double memoryUsageMB;
    double diskUsageGB;
    double networkUsageMBps;
    qint64 timestamp;
};

class PerformanceMonitor : public QObject
{
    Q_OBJECT

public:
    explicit PerformanceMonitor(QObject *parent = nullptr);
    ~PerformanceMonitor();

    void startMonitoring(int intervalMs = 1000);
    void stopMonitoring();
    void resetMetrics();
    
    PerformanceMetrics getCurrentMetrics() const;
    QList<PerformanceMetrics> getMetricsHistory() const;
    double getAverageCPUUsage() const;
    double getAverageMemoryUsage() const;
    double getPeakCPUUsage() const;
    double getPeakMemoryUsage() const;

signals:
    void metricsUpdated(const PerformanceMetrics &metrics);
    void highCPUUsage(double usage);
    void highMemoryUsage(double usage);

private slots:
    void updateMetrics();

private:
    void setupTimer();
    double getCPUUsage();
    double getMemoryUsage();
    double getDiskUsage();
    double getNetworkUsage();
    
    // Linux-specific methods
    double getLinuxCPUUsage();
    double getLinuxMemoryUsage();
    double getLinuxDiskUsage();
    double getLinuxNetworkUsage();
    
    // Windows-specific methods (if needed)
    double getWindowsCPUUsage();
    double getWindowsMemoryUsage();
    double getWindowsDiskUsage();
    double getWindowsNetworkUsage();
    
    QTimer *m_timer;
    QList<PerformanceMetrics> m_metricsHistory;
    PerformanceMetrics m_currentMetrics;
    
    // Previous values for rate calculations
    qint64 m_lastCPUIdle;
    qint64 m_lastCPUTotal;
    qint64 m_lastNetworkRx;
    qint64 m_lastNetworkTx;
    qint64 m_lastNetworkTime;
    
    // Thresholds
    double m_cpuThreshold;
    double m_memoryThreshold;
    
    // Maximum history size
    static const int MAX_HISTORY_SIZE = 1000;
};

#endif // PERFORMANCEMONITOR_H 