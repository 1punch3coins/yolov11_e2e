#include "algo_logger.hpp"

namespace algoLogger{
    void update_os_str(std::string& os_str) {
        std::time_t timestamp = std::time(nullptr);
        tm* tm_local = std::localtime(&timestamp);
        auto* p_std_out = std::cout.rdbuf();
        std::stringbuf buf;
        std::cout.rdbuf(&buf);
        std::cout << "[";
        std::cout << std::setw(2) << std::setfill('0') << 1 + tm_local->tm_mon << "/";
        std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_mday << "/";
        std::cout << std::setw(4) << std::setfill('0') << 1900 + tm_local->tm_year << "-";
        std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_hour << ":";
        std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_min << ":";
        std::cout << std::setw(2) << std::setfill('0') << tm_local->tm_sec << "] ";
        std::cout.rdbuf(p_std_out);
        os_str = buf.str();
    }
    const char* get_severity_str(const Severity& severity) {
        switch (severity) {
            case Severity::kError:          return "[E] ";
            case Severity::kWarning:        return "[W] ";
            case Severity::kInfo:           return "[I] ";
            case Severity::kVerbose:        return "[V] ";
            default: assert(0);             return "";
        }
    }

    void ColoredLogger::log(Severity const& severity, const char* subj, const char* msg) noexcept {
        update_os_str(os_str_);
        if (severity <= severity_) {
            if (severity == Severity::kWarning) {
                printf("%s\033[33m[%s] %s\033[0m\n", os_str_.c_str(), subj, msg);
                return;
            }
            if (severity == Severity::kError) {
                printf("%s\033[31m[%s] %s\033[0m\n", os_str_.c_str(), subj, msg);
                return;
            }
            if (severity == Severity::kInfo) {
                printf("%s\033[32m[%s] %s\033[0m\n", os_str_.c_str(), subj, msg);
                return;
            }
            printf("%s[%s] %s\n", os_str_.c_str(), subj, msg);
        }
    }
    void ColoredLogger::log(Severity const& severity, std::string const& subj, std::string const& msg) noexcept {
        update_os_str(os_str_);
        if (severity <= severity_) {
            if (severity == Severity::kWarning) {
                printf("%s\033[33m[%s] %s\033[0m\n", os_str_.c_str(), subj.c_str(), msg.c_str());
                return;
            }
            if (severity == Severity::kError) {
                printf("%s\033[31m[%s] %s\033[0m\n", os_str_.c_str(), subj.c_str(), msg.c_str());
                return;
            }
            if (severity == Severity::kInfo) {
                printf("%s\033[32m[%s] %s\033[0m\n", os_str_.c_str(), subj.c_str(), msg.c_str());
                return;
            }
            printf("%s[%s] %s\n", os_str_.c_str(), subj.c_str(), msg.c_str());
        }
    }

    void PlainLogger::log(Severity const& severity, const char* subj, const char* msg) noexcept {
        update_os_str(os_str_);
        if (severity <= severity_) {
            printf("%s%s[%s] %s\n", os_str_.c_str(), get_severity_str(severity), subj, msg);
        }
    }
    void PlainLogger::log(Severity const& severity, std::string const& subj, std::string const& msg) noexcept {
        update_os_str(os_str_);
        if (severity <= severity_) {
            printf("%s%s[%s] %s\n", os_str_.c_str(), get_severity_str(severity), subj.c_str(), msg.c_str());
        }
    }

#if USE_COLOR_LOGGER
    ColoredLogger clogger;  // used for screen output
    Logger* logger = &clogger;
#else
    PlainLogger plogger;    // used for file output
    Logger* logger = &plogger;
#endif
}