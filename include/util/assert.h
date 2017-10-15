#ifndef UTIL_ASSERT_H
#define UTIL_ASSERT_H

#include <stdexcept>

#include "config.h"

// Available strategies in case of assertion failure:
// 0. Throw an util::Assertion_error when assertion failed.
// 1. Calls std::terminate when assertion failed.
// 2. Do nothing. Assuming that an assertion will never fail can yield best performance.
#define ASSERT_STRATEGY_THROW 0
#define ASSERT_STRATEGY_TERMINATE 1
#define ASSERT_STRATEGY_ASSUME 2

// Default strategy for ASSERT is throw.
#ifndef ASSERT_STRATEGY
#   define ASSERT_STRATEGY ASSERT_STRATEGY_THROW
#endif

// Default strategy for contract violation is throw.
#ifndef CONTRACT_STRATEGY
#   define CONTRACT_STRATEGY ASSERT_STRATEGY_THROW
#endif

#define STRINGIFY_IMPL(x) #x
#define STRINGIFY(x) STRINGIFY_IMPL(x)

// Hint to the compiler the likely outcome of a condition for optimisation.
#ifdef __GNUC__
#   define LIKELY(cond) __builtin_expect(!!(cond), 1)
#   define UNLIKELY(cond) __builtin_expect(!!(cond), 0)
#else
#   define LIKELY(cond) (!!(cond))
#   define UNLIKELY(cond) (!!(cond))
#endif

// Hint to the compiler that the condition will always be true for optimisation.
#ifdef _MSC_VER
#   define ASSUME(cond) __assume(cond)
#elif defined(__clang__)
#   define ASSUME(cond) __builtin_assume(cond)
#elif defined(__GNUC__)
#   define ASSUME(cond) ((cond) ? static_cast<void>(0) : __builtin_unreachable())
#else
#   define ASSUME(cond) static_cast<void>(0)
#endif

namespace util {

struct Assertion_error: std::logic_error {
    explicit Assertion_error(const char* message) : std::logic_error(message) {}
};

}

#define ASSERT_IMPL_THROW(cond, type) \
    (LIKELY(cond) ? static_cast<void>(0) \
                  : throw util::Assertion_error(type " `" #cond "` failed at " __FILE__ ":" STRINGIFY(__LINE__)))

#define ASSERT_IMPL_TERMINATE(cond, type) (LIKELY(cond) ? static_cast<void>(0) : std::terminate())

#if ASSERT_STRATEGY == ASSERT_STRATEGY_THROW
#   define ASSERT(cond) ASSERT_IMPL_THROW(cond, "assertion")
#elif ASSERT_STRATEGY == ASSERT_STRATEGY_TERMINATE
#   define ASSERT(cond) ASSERT_IMPL_TERMINATE(cond, "assertion")
#else
#   define ASSERT(cond) ASSUME(cond)
#endif

#if CONTRACT_STRATEGY == ASSERT_STRATEGY_THROW
#   define PRECONDITION(cond) ASSERT_IMPL_THROW(cond, "precondition")
#elif CONTRACT_STRATEGY == ASSERT_STRATEGY_TERMINATE
#   define PRECONDITION(cond) ASSERT_IMPL_TERMINATE(cond, "precondition")
#else
#   define PRECONDITION(cond) ASSUME(cond)
#endif

#endif

