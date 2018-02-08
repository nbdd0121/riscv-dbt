#ifndef EMU_STATE_H
#define EMU_STATE_H

#include <memory>
#include <stdexcept>

#include "emu/typedef.h"

namespace riscv {

struct Context;

};

namespace emu {

struct State {
    std::unique_ptr<riscv::Context> context;

    // The program/data break of the address space. original_brk represents the initial brk from information gathered
    // in elf. Both values are set initially to original_brk by elf_loader, and original_brk should not be be changed.
    // A constraint original_brk <= brk must be satisified.
    reg_t original_brk;
    reg_t brk;
    reg_t heap_start;
    reg_t heap_end;

    // A flag to determine whether to trace all system calls. If true then all guest system calls will be logged.
    bool strace;

    // A flag to determine whether to print instruction out when it is decoded.
    bool disassemble;

    // A flag to determine whether instret should be updated precisely in binary translated code.
    bool no_instret;

    // Upper limit of number of blocks that can be inlined by IR DBT.
    int inline_limit;
};

// All parts of the emulator will share a global state. Originally global variable is avoided, but by doing so many
// objects need to hold a reference to the state object, which incurs unnecessary overhead and complexity.
// TODO: We will be shifting from the state struct to global variables gradually.

// A flag to determine whether correctness in case of segmentation fault should be dealt strictly.
extern bool strict_exception;

// Whether compilation performance counters should be enabled.
extern bool monitor_performance;

// Whether direct memory access or call to helper should be generated for guest memory access.
extern bool no_direct_memory_access;

// This is not really an error. However it shares some properties with an exception, as it needs to break out from
// any nested controls and stop executing guest code.
struct Exit_control: std::runtime_error {
    uint8_t exit_code;
    Exit_control(uint8_t exit_code): std::runtime_error { "exit" }, exit_code {exit_code} {}
};

reg_t load_elf(const char *filename, State& state);

}

#endif
