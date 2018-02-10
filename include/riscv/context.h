#ifndef RISCV_STATE_H
#define RISCV_STATE_H

#include "main/executor.h"
#include "riscv/typedef.h"

namespace emu {
    class Mmu;
    class State;
}

namespace riscv {

// This class represent the state of a single hard (i.e. hardware thread).
struct Context {
    // registers[0] is reserved and maybe used for other purpose later
    reg_t registers[32];
    freg_t fp_registers[32];
    reg_t pc;
    reg_t instret;

    reg_t fcsr;

    // For load-reserved
    reg_t lr;

    // These are not part of hart state, but reference to the global system states. Technically shared memory is also
    // global system state, but a pointer to MMU is placed here since it is frequently accessed.
    emu::State *state;
    Executor *executor;
};

class Instruction;
void step(Context *context, Instruction inst);

} // riscv

#endif
