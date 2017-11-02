#ifndef EMU_STATE_H
#define EMU_STATE_H

#include <memory>

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

    // The exit code of the program. Set when the guest program calls syscall exit.
    int exit_code;

    // A flag to determine whether to trace all system calls. If true then all guest system calls will be logged.
    bool strace;

    // A flag to determine whether to print instruction out when it is decoded.
    bool disassemble;
};

}

#endif
