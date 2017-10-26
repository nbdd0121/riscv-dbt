#ifndef UTIL_FORMAT_H
#define UTIL_FORMAT_H

#include <ostream>

namespace util {

template<typename T>
struct Formatter {
    static void format(std::ostream& stream, const T& value) {
        stream << value;
    }
};

namespace internal {

struct Bound_formatter {
    using Format_function = void (*)(std::ostream& stream, const void *value);

    Format_function formatter;
    const void *value;

    template<typename T>
    Bound_formatter(const T& reference) {
        formatter = [](std::ostream& stream, const void *value) {
            Formatter<T>::format(stream, *reinterpret_cast<const T*>(value));
        };
        value = reinterpret_cast<const void*>(&reference);
    }
    
    void format(std::ostream& stream) {
        formatter(stream, value);
    }
};

void format_impl(std::ostream& stream, const char *format, Bound_formatter* view, size_t size);

}

template<typename... Args>
void format(std::ostream& stream, const char *format, const Args&... args) {
    util::internal::Bound_formatter list[] = { args... };
    format_impl(stream, format, list, sizeof...(args));
}

}

#endif
