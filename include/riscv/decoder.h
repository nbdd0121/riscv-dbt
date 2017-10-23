#ifndef RISCV_DECODER_H
#define RISCV_DECODER_H

#include <cstdint>

#include "riscv/typedef.h"

namespace riscv {

class Instruction;

class Decoder {
    uint32_t bits_;

public:
    Decoder(): bits_ {0} {}
    Decoder(uint32_t bits): bits_ {bits} {}

    Instruction decode() const;
};

} // riscv

#endif
