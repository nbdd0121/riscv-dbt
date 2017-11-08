#ifndef MAIN_DBT_H
#define MAIN_DBT_H

#include <cstdint>
#include <cstddef>
#include <memory>
#include <unordered_map>

#include "emu/typedef.h"
#include "util/code_buffer.h"

namespace emu {
struct State;
};

namespace riscv {
    struct Context;
}

namespace util {
class Code_buffer;
};

class Dbt_runtime {
private:
    emu::State& state_;

    // The following two fields are for hot direct-mapped instruction cache that contains recently executed code.
    std::unique_ptr<emu::reg_t[]> icache_tag_;
    std::unique_ptr<std::byte*[]> icache_;

    // The "slow" instruction cache that contains all code that are compiled previously.
    std::unordered_map<emu::reg_t, util::Code_buffer> inst_cache_;

    void compile(emu::reg_t);

public:
    Dbt_runtime(emu::State& state);
    void step(riscv::Context& context);

    friend class Dbt_compiler;
};

#endif
