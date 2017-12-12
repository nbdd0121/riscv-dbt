#ifndef MAIN_IR_EVALUATOR_H
#define MAIN_IR_EVALUATOR_H

#include <unordered_map>

#include "emu/typedef.h"
#include "ir/instruction.h"

namespace emu {
struct State;
}

namespace riscv {
struct Context;
}

class Ir_evaluator {
private:
    emu::State& state_;
    std::unordered_map<emu::reg_t, ir::Graph> inst_cache_;

public:
    Ir_evaluator(emu::State& state) noexcept;
    void step(riscv::Context& context);
};

#endif
