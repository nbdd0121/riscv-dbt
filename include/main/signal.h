#ifndef MAIN_SIGNAL_H
#define MAIN_SIGNAL_H

#include <stdexcept>

class Segv_exception: public std::runtime_error {
public:
    Segv_exception(): std::runtime_error {"segmentation fault"} {}
};

class Fpe_exception: public std::runtime_error {
public:
    Fpe_exception(): std::runtime_error {"floating point exception"} {}
};

void setup_fault_handler();

#endif
