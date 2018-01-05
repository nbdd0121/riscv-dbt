#ifndef MAIN_IR_DBT_H
#define MAIN_IR_DBT_H

#include <memory>
#include <unordered_map>

#include "emu/typedef.h"
#include "ir/node.h"
#include "util/code_buffer.h"

namespace emu {
struct State;
}

namespace riscv {
struct Context;
}

struct Ir_block;

class Ir_dbt {
private:
    emu::State& state_;

     // The following two fields are for hot direct-mapped instruction cache that contains recently executed code.
    std::unique_ptr<emu::reg_t[]> icache_tag_;
    std::unique_ptr<std::byte*[]> icache_;

    // The "slow" instruction cache that contains all code that are compiled previously.
    std::unordered_map<emu::reg_t, std::unique_ptr<Ir_block>> inst_cache_;

    void* _code_ptr_to_patch = nullptr;

public:
    Ir_dbt(emu::State& state) noexcept;
    ~Ir_dbt();
    void step(riscv::Context& context);
    void compile(emu::reg_t pc);
};

#endif
