#ifndef MAIN_SIGNAL_H
#define MAIN_SIGNAL_H

class Segv_exception {};
class Fpe_exception {};

void setup_fault_handler();

#endif
