#ifndef RISCV_DECODER_H
#define RISCV_DECODER_H

#include <cstdint>

#include "riscv/typedef.h"

namespace emu {
class State;
}

namespace riscv {

class Instruction;
struct Basic_block;

class Decoder {
    emu::State *state_;
    reg_t pc_;

public:
    static Instruction decode(uint32_t bits);

public:
    Decoder(emu::State *state): state_{state}, pc_{0} {}
    Decoder(emu::State *state, reg_t pc): state_{state}, pc_{pc} {}

    reg_t pc() const { return pc_; }
    void pc(reg_t pc) { pc_ = pc; }

    Instruction decode_instruction();
    Basic_block decode_basic_block();
};

} // riscv

#endif
