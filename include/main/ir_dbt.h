#ifndef MAIN_IR_DBT_H
#define MAIN_IR_DBT_H

#include <memory>
#include <unordered_map>

#include "emu/typedef.h"
#include "ir/node.h"
#include "main/executor.h"
#include "util/code_buffer.h"

namespace riscv {
struct Context;
}

struct Ir_block;

class Ir_dbt final: public Executor {
private:
     // The following two fields are for hot direct-mapped instruction cache that contains recently executed code.
    std::unique_ptr<emu::reg_t[]> icache_tag_;
    std::unique_ptr<std::byte*[]> icache_;

    // The "slow" instruction cache that contains all code that are compiled previously.
    std::unordered_map<emu::reg_t, std::unique_ptr<Ir_block>> inst_cache_;

    int64_t total_compilation_time = 0;
    size_t total_block_compiled = 0;

    std::byte* _code_ptr_to_patch = nullptr;
    bool _need_cache_flush = false;

public:
    Ir_dbt() noexcept;
    ~Ir_dbt();
    void step(riscv::Context& context);
    void decode(emu::reg_t pc);
    void compile(emu::reg_t pc);
    virtual void flush_cache() override;
};

#endif
