#ifndef EMU_STATE_H
#define EMU_STATE_H

#include <memory>
#include <stdexcept>

#include "emu/typedef.h"

namespace riscv {

struct Context;

};

namespace emu {

class Mmu;

struct State {
    std::unique_ptr<riscv::Context> context;
    std::unique_ptr<Mmu> mmu;

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

    // A flag to determine whether correctness in case of segmentation fault should be dealt strictly.
    bool strict_exception;

    // Upper limit of number of blocks that can be inlined by IR DBT.
    int inline_limit;
};

// This is not really an error. However it shares some properties with an exception, as it needs to break out from
// any nested controls and stop executing guest code.
struct Exit_control: std::runtime_error {
    uint8_t exit_code;
    Exit_control(uint8_t exit_code): std::runtime_error { "exit" }, exit_code {exit_code} {}
};

reg_t load_elf(const char *filename, State& mmu);

}

#endif
