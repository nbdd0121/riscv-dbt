#ifndef RISCV_FRONTEND_H
#define RISCV_FRONTEND_H

#include "ir/node.h"

namespace emu {
struct State;
}

namespace riscv {

class Basic_block;

ir::Graph compile(emu::State& state, const Basic_block& block);

} // riscv

#endif
