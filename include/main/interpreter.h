#ifndef MAIN_INTERPRETER_H
#define MAIN_INTERPRETER_H

#include <unordered_map>

#include "emu/typedef.h"
#include "main/executor.h"
#include "riscv/basic_block.h"

namespace emu {
struct State;
}

namespace riscv {
struct Context;
}

class Interpreter: public Executor {
private:
    emu::State& state_;
    std::unordered_map<emu::reg_t, riscv::Basic_block> inst_cache_;

public:
    Interpreter(emu::State& state) noexcept;
    ~Interpreter();
    void step(riscv::Context& context);
    virtual void flush_cache() override;
};

#endif
