#include <iostream>
#include <iomanip>
#include <ctime>
#include <assert.h>
#include "trt_logger.hpp"

namespace trtLogger{
    const char* get_severity_str(Severity& severity) {
        switch (severity) {
            case Severity::kINTERNAL_ERROR: return "[F] ";
            case Severity::kERROR:          return "[E] ";
            case Severity::kWARNING:        return "[W] ";
            case Severity::kINFO:           return "[I] ";
            case Severity::kVERBOSE:        return "[V] ";
            default: assert(0);             return "";
        }
    }
    void update_os_str(std::string& os_str) {
        std::time_t timestamp = std::time(nullptr);
        tm* tm_local = std::localtime(&timestamp);
        auto* pStdBuf = std::cout.rdbuf();
        std::stringbuf buf;
        std::cout.rdbuf(&buf);
        std::cout << "[";
        std::cout << std::setw(2) << std::setfill('0') << 1 + tm_local->tm_mon << "/";
        std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_mday << "/";
        std::cout << std::setw(4) << std::setfill('0') << 1900 + tm_local->tm_year << "-";
        std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_hour << ":";
        std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_min << ":";
        std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_sec << "] ";
        std::cout.rdbuf(pStdBuf);
        os_str = buf.str();
    }

    void ColoredLogger::log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept {
        update_os_str(os_str_);
        if (severity <= severity_) {
            if (severity == Severity::kWARNING) {
                printf("%s\033[33m%s[TRT] %s\033[0m\n", os_str_.c_str(), get_severity_str(severity), msg);
            }
            else if (severity <= Severity::kERROR) {
                printf("%s\033[31m%s[TRT] %s\033[0m\n", os_str_.c_str(), get_severity_str(severity), msg);
            }
            else {
                printf("%s%s[TRT] %s\n", os_str_.c_str(), get_severity_str(severity), msg);
            }
        }
    }
    ColoredLogger clogger{Severity::kINFO};
}