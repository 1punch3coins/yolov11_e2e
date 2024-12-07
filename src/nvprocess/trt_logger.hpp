#pragma once
#include <NvInfer.h>
#include <string>

#define SET_ERROR_INFO(ptr, num, ...) {     \
    std::string format;                     \
    for (unsigned i = 0; i < num-1; i++) {  \
        format += "%s ";                    \
    }                                       \
    format += "%s";                         \
    sprintf(ptr, format.c_str(), __VA_ARGS__);      \
}

#define LOG_TRT_ERROR(logger, num, ...) {   \
    char buf[sizeof(__VA_ARGS__)];          \
}

namespace trtLogger{
    using Severity = nvinfer1::ILogger::Severity;
    const char* get_severity_str(Severity& severity);
    void update_os_str(std::string& os_str);

    class ColoredLogger : public nvinfer1::ILogger {
    public:
        ColoredLogger(Severity severity):
            severity_(severity)
        {}

        virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override;

        void resetSeverity(Severity severity) {
            severity_ = severity;
        }

        ColoredLogger& getTRTLogger() noexcept {
            return *this;
        }
    
    private:
        std::string os_str_;
        Severity severity_;
    };

    extern ColoredLogger clogger;
}
