#ifndef ALGO_LOGGER_HPP_
#define ALGO_LOGGER_HPP_
#include <stdio.h>
#include <ctime>
#include <iomanip>
#include <ostream>
#include <iostream>
#include <string>
#include <assert.h>

// macros are not restricted by namespaces
#define APPEND_ERROR_LOC(ptr, info) {                             \
    sprintf(ptr, "%s, error at: %s:%d", info, __FILE__, __LINE__);\
}

#define SET_TIMING_INFO(ptr, pre, duration, post) {    \
    sprintf(ptr, "%s: %f %s", pre, duration, post);    \
}

#define LOG_ERROR(logger, ident, info) {      \
    char buf[sizeof(ident)+sizeof(info)+100];                \
    APPEND_ERROR_LOC(buf, info);                             \
    logger->log(algoLogger::Severity::kError, ident, buf);   \
}

#define LOG_ERROR_RETURN(logger, ident, info) { \
    LOG_ERROR(logger, ident, info);             \
    return 1;                                   \
}

#define LOG_INFO(logger, ident, info) {      \
    logger->log(algoLogger::Severity::kInfo, ident, info);   \
}

namespace algoLogger{
    enum class Severity: int32_t {
        kError   = 0,
        kWarning = 1,
        kInfo    = 2,
        kVerbose = 3
    };
    void update_os_str(std::string& os_str);
    const char* get_severity_str(const Severity& severity);

    class Logger {
    public:
        Logger(Severity severity = Severity::kInfo):
            severity_(severity)
        {}
        virtual ~Logger() = default;
        virtual void log(Severity const& severity, const char* subj, const char* msg) noexcept = 0;
        virtual void log(Severity const& severity, std::string const& subj, std::string const& msg) noexcept = 0;
        void resetSeverity(Severity severity) {severity_ = severity;}
        Logger* getLogger() noexcept {return this;}
    protected:
        Severity severity_;
        std::string os_str_;
    };

    class ColoredLogger : public Logger {
    public:
        void log(Severity const& severity, const char* subj, const char* msg) noexcept override;
        void log(Severity const& severity, std::string const& subj, std::string const& msg) noexcept override;
        ColoredLogger& getALGOLogger() noexcept {return *this;}
    };

    class PlainLogger : public Logger{
    public:
        void log(Severity const& severity, const char* subj, const char* msg) noexcept override;
        void log(Severity const& severity, std::string const& subj, std::string const& msg) noexcept override;
        PlainLogger& getALGOLogger() noexcept {return *this;}
    };

    extern Logger* logger;
}

#endif