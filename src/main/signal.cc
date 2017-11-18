#include <csignal>
#include <cstring>

#include "main/signal.h"

namespace {

void handle_fault(int sig) {
    sigset_t x;
    sigemptyset(&x);
    sigaddset(&x, sig);
    sigprocmask(SIG_UNBLOCK, &x, nullptr);
    if (sig == SIGSEGV) {
        throw Segv_exception {};
    } else {
        throw Fpe_exception {};
    }
}

}

void setup_fault_handler() {
    struct sigaction act;
    memset (&act, 0, sizeof(act));
    act.sa_handler = handle_fault;
    sigaction(SIGSEGV, &act, NULL);
    sigaction(SIGFPE, &act, NULL);
}
