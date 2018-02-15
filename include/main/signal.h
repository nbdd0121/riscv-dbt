#ifndef MAIN_SIGNAL_H
#define MAIN_SIGNAL_H

#include <stdexcept>

struct Segv_exception: std::runtime_error {
    Segv_exception(): std::runtime_error {"segmentation fault"} {}
};

void setup_fault_handler();

#endif
