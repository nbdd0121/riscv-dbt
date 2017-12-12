#include "emu/state.h"
#include "ir/pass.h"
#include "main/ir_evaluator.h"
#include "riscv/basic_block.h"
#include "riscv/context.h"
#include "riscv/decoder.h"
#include "riscv/disassembler.h"
#include "riscv/frontend.h"
#include "riscv/instruction.h"
#include "riscv/opcode.h"
#include "util/assert.h"
#include "util/format.h"

Ir_evaluator::Ir_evaluator(emu::State& state) noexcept: state_{state} {

}

void Ir_evaluator::step(riscv::Context& context) {
    emu::reg_t pc = context.pc;
    ir::Graph& graph = inst_cache_[pc];

    if (UNLIKELY(!graph.root())) {
        riscv::Decoder decoder {&state_, pc};
        riscv::Basic_block basic_block = decoder.decode_basic_block();

        graph = riscv::compile(basic_block);
        ir::pass::Register_access_elimination{66}.run(graph);
        ir::pass::Local_value_numbering{}.run(graph);

        if (context.state->disassemble) {
            ir::pass::Dot_printer{}.run(graph);
        }

        graph.garbage_collect();
    }

    ir::pass::Evaluator{&context}.run(graph);
}
