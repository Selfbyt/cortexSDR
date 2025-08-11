#include "PerformanceMonitor.h"
#include <QProcess>
#include <QTextStream>
#include <QFile>
#include <QDir>
#include <QDebug>

PerformanceMonitor::PerformanceMonitor(QObject *parent)
    : QObject(parent)
    , m_timer(nullptr)
    , m_lastCPUIdle(0)
    , m_lastCPUTotal(0)
    , m_lastNetworkRx(0)
    , m_lastNetworkTx(0)
    , m_lastNetworkTime(0)
    , m_cpuThreshold(80.0)
    , m_memoryThreshold(90.0)
{
    setupTimer();
}

PerformanceMonitor::~PerformanceMonitor()
{
    stopMonitoring();
}

void PerformanceMonitor::startMonitoring(int intervalMs)
{
    if (m_timer) {
        m_timer->start(intervalMs);
    }
}

void PerformanceMonitor::stopMonitoring()
{
    if (m_timer) {
        m_timer->stop();
    }
}

void PerformanceMonitor::resetMetrics()
{
    m_metricsHistory.clear();
    m_currentMetrics = PerformanceMetrics();
}

PerformanceMetrics PerformanceMonitor::getCurrentMetrics() const
{
    return m_currentMetrics;
}

QList<PerformanceMetrics> PerformanceMonitor::getMetricsHistory() const
{
    return m_metricsHistory;
}

double PerformanceMonitor::getAverageCPUUsage() const
{
    if (m_metricsHistory.isEmpty()) return 0.0;
    
    double sum = 0.0;
    for (const auto &metrics : m_metricsHistory) {
        sum += metrics.cpuUsage;
    }
    return sum / m_metricsHistory.size();
}

double PerformanceMonitor::getAverageMemoryUsage() const
{
    if (m_metricsHistory.isEmpty()) return 0.0;
    
    double sum = 0.0;
    for (const auto &metrics : m_metricsHistory) {
        sum += metrics.memoryUsageMB;
    }
    return sum / m_metricsHistory.size();
}

double PerformanceMonitor::getPeakCPUUsage() const
{
    if (m_metricsHistory.isEmpty()) return 0.0;
    
    double peak = 0.0;
    for (const auto &metrics : m_metricsHistory) {
        peak = qMax(peak, metrics.cpuUsage);
    }
    return peak;
}

double PerformanceMonitor::getPeakMemoryUsage() const
{
    if (m_metricsHistory.isEmpty()) return 0.0;
    
    double peak = 0.0;
    for (const auto &metrics : m_metricsHistory) {
        peak = qMax(peak, metrics.memoryUsageMB);
    }
    return peak;
}

void PerformanceMonitor::updateMetrics()
{
    PerformanceMetrics metrics;
    metrics.timestamp = QDateTime::currentMSecsSinceEpoch();
    
    // Get current metrics
    metrics.cpuUsage = getCPUUsage();
    metrics.memoryUsageMB = getMemoryUsage();
    metrics.diskUsageGB = getDiskUsage();
    metrics.networkUsageMBps = getNetworkUsage();
    
    // Update current metrics
    m_currentMetrics = metrics;
    
    // Add to history
    m_metricsHistory.append(metrics);
    
    // Limit history size
    if (m_metricsHistory.size() > MAX_HISTORY_SIZE) {
        m_metricsHistory.removeFirst();
    }
    
    // Check thresholds
    if (metrics.cpuUsage > m_cpuThreshold) {
        emit highCPUUsage(metrics.cpuUsage);
    }
    
    if (metrics.memoryUsageMB > m_memoryThreshold) {
        emit highMemoryUsage(metrics.memoryUsageMB);
    }
    
    // Emit update signal
    emit metricsUpdated(metrics);
}

void PerformanceMonitor::setupTimer()
{
    m_timer = new QTimer(this);
    connect(m_timer, &QTimer::timeout, this, &PerformanceMonitor::updateMetrics);
}

double PerformanceMonitor::getCPUUsage()
{
#ifdef Q_OS_LINUX
    return getLinuxCPUUsage();
#elif defined(Q_OS_WIN)
    return getWindowsCPUUsage();
#else
    return 0.0;
#endif
}

double PerformanceMonitor::getMemoryUsage()
{
#ifdef Q_OS_LINUX
    return getLinuxMemoryUsage();
#elif defined(Q_OS_WIN)
    return getWindowsMemoryUsage();
#else
    return 0.0;
#endif
}

double PerformanceMonitor::getDiskUsage()
{
#ifdef Q_OS_LINUX
    return getLinuxDiskUsage();
#elif defined(Q_OS_WIN)
    return getWindowsDiskUsage();
#else
    return 0.0;
#endif
}

double PerformanceMonitor::getNetworkUsage()
{
#ifdef Q_OS_LINUX
    return getLinuxNetworkUsage();
#elif defined(Q_OS_WIN)
    return getWindowsNetworkUsage();
#else
    return 0.0;
#endif
}

double PerformanceMonitor::getLinuxCPUUsage()
{
    QFile file("/proc/stat");
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return 0.0;
    }
    
    QTextStream in(&file);
    QString line = in.readLine();
    file.close();
    
    if (line.startsWith("cpu ")) {
        QStringList parts = line.split(" ", Qt::SkipEmptyParts);
        if (parts.size() >= 5) {
            qint64 user = parts[1].toLongLong();
            qint64 nice = parts[2].toLongLong();
            qint64 system = parts[3].toLongLong();
            qint64 idle = parts[4].toLongLong();
            qint64 iowait = parts.size() > 5 ? parts[5].toLongLong() : 0;
            qint64 irq = parts.size() > 6 ? parts[6].toLongLong() : 0;
            qint64 softirq = parts.size() > 7 ? parts[7].toLongLong() : 0;
            
            qint64 total = user + nice + system + idle + iowait + irq + softirq;
            qint64 nonIdle = user + nice + system + irq + softirq;
            
            if (m_lastCPUTotal > 0) {
                qint64 totalDiff = total - m_lastCPUTotal;
                qint64 idleDiff = idle - m_lastCPUIdle;
                
                if (totalDiff > 0) {
                    double cpuUsage = 100.0 * (totalDiff - idleDiff) / totalDiff;
                    m_lastCPUIdle = idle;
                    m_lastCPUTotal = total;
                    return cpuUsage;
                }
            }
            
            m_lastCPUIdle = idle;
            m_lastCPUTotal = total;
        }
    }
    
    return 0.0;
}

double PerformanceMonitor::getLinuxMemoryUsage()
{
    QFile file("/proc/meminfo");
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return 0.0;
    }
    
    QTextStream in(&file);
    qint64 totalMem = 0;
    qint64 freeMem = 0;
    qint64 availableMem = 0;
    
    while (!in.atEnd()) {
        QString line = in.readLine();
        if (line.startsWith("MemTotal:")) {
            totalMem = line.split(" ", Qt::SkipEmptyParts)[1].toLongLong();
        } else if (line.startsWith("MemFree:")) {
            freeMem = line.split(" ", Qt::SkipEmptyParts)[1].toLongLong();
        } else if (line.startsWith("MemAvailable:")) {
            availableMem = line.split(" ", Qt::SkipEmptyParts)[1].toLongLong();
        }
    }
    file.close();
    
    if (totalMem > 0) {
        qint64 usedMem = totalMem - (availableMem > 0 ? availableMem : freeMem);
        return (usedMem * 1024.0) / (1024.0 * 1024.0); // Convert to MB
    }
    
    return 0.0;
}

double PerformanceMonitor::getLinuxDiskUsage()
{
    QProcess process;
    process.start("df", QStringList() << "/" << "--output=size,used");
    process.waitForFinished();
    
    QString output = process.readAllStandardOutput();
    QStringList lines = output.split("\n", Qt::SkipEmptyParts);
    
    if (lines.size() >= 2) {
        QStringList parts = lines[1].split(" ", Qt::SkipEmptyParts);
        if (parts.size() >= 2) {
            qint64 total = parts[0].toLongLong();
            qint64 used = parts[1].toLongLong();
            
            if (total > 0) {
                return (used * 1024.0) / (1024.0 * 1024.0 * 1024.0); // Convert to GB
            }
        }
    }
    
    return 0.0;
}

double PerformanceMonitor::getLinuxNetworkUsage()
{
    QFile file("/proc/net/dev");
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return 0.0;
    }
    
    QTextStream in(&file);
    qint64 totalRx = 0;
    qint64 totalTx = 0;
    
    // Skip header lines
    in.readLine();
    in.readLine();
    
    while (!in.atEnd()) {
        QString line = in.readLine();
        QStringList parts = line.split(" ", Qt::SkipEmptyParts);
        
        if (parts.size() >= 10) {
            // Skip loopback interface
            if (!parts[0].contains("lo:")) {
                totalRx += parts[1].toLongLong();
                totalTx += parts[9].toLongLong();
            }
        }
    }
    file.close();
    
    qint64 currentTime = QDateTime::currentMSecsSinceEpoch();
    
    if (m_lastNetworkTime > 0) {
        qint64 timeDiff = currentTime - m_lastNetworkTime;
        if (timeDiff > 0) {
            qint64 rxDiff = totalRx - m_lastNetworkRx;
            qint64 txDiff = totalTx - m_lastNetworkTx;
            qint64 totalDiff = rxDiff + txDiff;
            
            double mbps = (totalDiff * 8.0) / (timeDiff * 1000.0 * 1024.0 * 1024.0); // Convert to MB/s
            
            m_lastNetworkRx = totalRx;
            m_lastNetworkTx = totalTx;
            m_lastNetworkTime = currentTime;
            
            return mbps;
        }
    }
    
    m_lastNetworkRx = totalRx;
    m_lastNetworkTx = totalTx;
    m_lastNetworkTime = currentTime;
    
    return 0.0;
}

double PerformanceMonitor::getWindowsCPUUsage()
{
    // TODO: Implement Windows CPU usage monitoring
    return 0.0;
}

double PerformanceMonitor::getWindowsMemoryUsage()
{
    // TODO: Implement Windows memory usage monitoring
    return 0.0;
}

double PerformanceMonitor::getWindowsDiskUsage()
{
    // TODO: Implement Windows disk usage monitoring
    return 0.0;
}

double PerformanceMonitor::getWindowsNetworkUsage()
{
    // TODO: Implement Windows network usage monitoring
    return 0.0;
} 