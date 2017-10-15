#ifndef UTIL_MEMORY_H
#define UTIL_MEMORY_H

#include <cstring>

namespace util {

namespace internal {

// A helper class to be able to let return value of interpret_as to be used as if it's a reference.
template<typename T>
struct interpret_as_proxy {
    void* pointer_;

    interpret_as_proxy(void* pointer) noexcept: pointer_{pointer} {}

    // Disallow the proxy to be copied or removed. Note that pre-C++17 this will also disallow `auto v = <proxy>`,
    // but post-C++17 due to introduction of guaranteed copy elision, this no longer stops user from doing it.
    // but doing so is still discouraged.
    interpret_as_proxy(const interpret_as_proxy&) = delete;
    interpret_as_proxy(interpret_as_proxy&&) = delete;

    // Return value is intentionally set to void to discourage chained assignments.
    void operator =(const T& value) noexcept {
        memcpy(pointer_, &value, sizeof(T));
        return *this;
    }

    void operator =(const interpret_as_proxy&) = delete;
    void operator =(interpret_as_proxy&&) = delete;

    operator T() noexcept {
        T ret;
        memcpy(&ret, pointer_, sizeof(T));
        return ret;
    }
};

}

// Accessing a byte array as another type will violate the strict aliasing rule. Considering that operation is very
// common, interpret_as is implemented to maximise standard conformance.
// The returned proxy should be either immediated casted, or immediately assigned.
template<typename T>
internal::interpret_as_proxy<T> interpret_as(void *memory) {
    return {memory};
}

}

#endif
