#include <cstring>

#include "emu/state.h"
#include "emu/unwind.h"
#include "ir/pass.h"
#include "main/ir_dbt.h"
#include "main/signal.h"
#include "riscv/basic_block.h"
#include "riscv/context.h"
#include "riscv/decoder.h"
#include "riscv/disassembler.h"
#include "riscv/frontend.h"
#include "riscv/instruction.h"
#include "riscv/opcode.h"
#include "util/assert.h"
#include "util/format.h"
#include "util/memory.h"
#include "x86/backend.h"

// Declare the exception handling registration functions.
extern "C" void __register_frame(void*);
extern "C" void __deregister_frame(void*);

// Denotes a translated block.
struct Ir_block {

    // Translated code.
    util::Code_buffer code;

    // Graph representing the basic block.
    ir::Graph graph;

    // Exception handling frame
    std::unique_ptr<uint8_t[]> cie;

    ~Ir_block() {
        if (cie) {
            __deregister_frame(cie.get());
        }
    }
};

_Unwind_Reason_Code ir_dbt_personality(
    [[maybe_unused]] int version,
    [[maybe_unused]] _Unwind_Action actions,
    [[maybe_unused]] uint64_t exception_class,
    [[maybe_unused]] struct _Unwind_Exception *exception_object,
    [[maybe_unused]] struct _Unwind_Context *context
) {
    return _URC_CONTINUE_UNWIND;
}

static void generate_eh_frame(Ir_block& block) {
    // TODO: Create an dwarf generation to replace this hard-coded template.
    static const unsigned char cie_template[] = {
        // CIE
        // Length
        0x1C, 0x00, 0x00, 0x00,
        // CIE
        0x00, 0x00, 0x00, 0x00,
        // Version
        0x01,
        // Augmentation string
        'z', 'P', 'L', 0,
        // Instruction alignment factor = 1
        0x01,
        // Data alignment factor = -8
        0x78,
        // Return register number
        0x10,
        // Augmentation data
        0x0A, // Data for z
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // abs format, personality routine
        0x00, // abs format for LSDA
        // Instructions
        // def_cfa(rsp, 8)
        0x0c, 0x07, 0x08,
        // offset(rsp, cfa-8)
        0x90, 0x01,
        // Padding

        // FDE
        // Length
        0x24, 0x00, 0x00, 0x00,
        // CIE Pointer
        0x24, 0x00, 0x00, 0x00,
        // Initial location
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        // Augumentation data
        0x8,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // LSDA
        // advance_loc(1)
        0x41,
        // def_cfa_offset(16)
        0x0E, 0x10,
        // offset(rbp, cfa-16)
        0x86, 0x02,
        // Padding
        0x00, 0x00,

        0x00, 0x00, 0x00, 0x00
    };

    block.cie = std::make_unique<uint8_t[]>(sizeof(cie_template));
    uint8_t *cie = block.cie.get();

    memcpy(cie, cie_template, sizeof(cie_template));
    util::write_as<uint64_t>(cie + 0x12, reinterpret_cast<uint64_t>(ir_dbt_personality));
    util::write_as<uint64_t>(cie + 0x28, reinterpret_cast<uint64_t>(block.code.data()));
    util::write_as<uint64_t>(cie + 0x30, 4096);
    util::write_as<uint64_t>(cie + 0x39, 0);

    __register_frame(cie);
}

Ir_dbt::Ir_dbt(emu::State& state) noexcept: state_{state} {
    icache_tag_ = std::make_unique<emu::reg_t[]>(4096);
    icache_ = std::make_unique<std::byte*[]>(4096);
    for (size_t i = 0; i < 4096; i++) {
        icache_tag_[i] = 0;
    }
}

Ir_dbt::~Ir_dbt() {}

void Ir_dbt::step(riscv::Context& context) {
    const emu::reg_t pc = context.pc;
    const ptrdiff_t tag = (pc >> 1) & 4095;

    // If the cache misses, compile the current block.
    if (UNLIKELY(icache_tag_[tag] != pc)) {
        compile(pc);
    }

    // The return value is the address to patch.
    auto func = reinterpret_cast<std::byte*(*)(riscv::Context&)>(icache_[tag]);
    ASSERT(func);

    if (_code_ptr_to_patch) {
        // Patch the trampoline.
        // mov rax, i64 => 48 B8 i64
        // jmp rax => FF E0
        util::write_as<uint16_t>(_code_ptr_to_patch, 0xB848);
        util::write_as<uint64_t>(_code_ptr_to_patch + 2, reinterpret_cast<uint64_t>(icache_[tag]) + 4);
        util::write_as<uint16_t>(_code_ptr_to_patch + 10, 0xE0FF);
    }

    _code_ptr_to_patch = func(context);
}

void Ir_dbt::decode(emu::reg_t pc) {
    auto& block_ptr = inst_cache_[pc];

    if (!block_ptr) {
        block_ptr = std::make_unique<Ir_block>();

        ir::Graph& graph = block_ptr->graph;
        riscv::Decoder decoder {&state_, pc};
        riscv::Basic_block basic_block = decoder.decode_basic_block();

        // Frontend stages.
        graph = riscv::compile(state_, basic_block);

        // Optimisation passes.
        ir::pass::Register_access_elimination{66, state_.strict_exception}.run(graph);
        ir::pass::Local_value_numbering{}.run(graph);

        // Clean up memory.
        graph.garbage_collect();
    }
}

void Ir_dbt::compile(emu::reg_t pc) {
    const ptrdiff_t tag = (pc >> 1) & 4095;

    decode(pc);
    auto& block_ptr = inst_cache_[pc];
    ASSERT(block_ptr);

    if (block_ptr->code.empty()) {
        block_ptr->code.reserve(4096);

        ir::Graph& graph = block_ptr->graph;
        ir::Graph graph_for_codegen = graph.clone();

        // A map between emulated pc and entry point in the graph.
        std::unordered_map<emu::reg_t, ir::Node*> block_map;
        block_map[pc] = *graph_for_codegen.entry()->value(0).references().begin();

        int counter = 0;
        bool changed = true;

        // Keep inlining until no changes are made.
        while (changed) {
            changed = false;
            for (auto operand: graph_for_codegen.exit()->operands()) {
                ir::Value target_pc_value = ir::pass::Register_access_elimination::get_tail_jmp_pc(operand, 64);

                // We can inline tail jump.
                if (target_pc_value && target_pc_value.is_const()) {
                    auto target_pc = target_pc_value.const_value();
                    if (!target_pc) continue;

                    auto block = block_map[target_pc];

                    if (block) {

                        // Add a new entry edge to the block.
                        block->operand_add(operand);

                        // Note that now operand is referenced by `block` and the end node. This is a special as
                        // usually control can only be referenced by one node. The edge from end to operand is called
                        // keepalive edge which prevents GC from reclaiming endless loops.

                    } else if (counter < state_.inline_limit) {

                        // To avoid spending too much time inlining all possible branches, we set an upper limit.

                        // Decode and clone the graph of the block to be inlined.
                        decode(target_pc);
                        ir::Graph graph_to_inline = inst_cache_[target_pc]->graph.clone();

                        // Store the entry point of the inlined graph.
                        block_map[target_pc] = *graph_to_inline.entry()->value(0).references().begin();

                        if (state_.disassemble) {
                            util::log("inline {:x} to {:x}\n", target_pc, pc);
                        }

                        // Inline the graph. Note that the iterator is invalidated so we need to break.
                        graph_for_codegen.inline_graph(operand, std::move(graph_to_inline));

                        changed = true;
                        counter++;
                        break;
                    }
                }
            }
        }

        // Optimisation passes.
        ir::pass::Block_combine{}.run(graph_for_codegen);
        ir::pass::Register_access_elimination{66, state_.strict_exception}.run(graph_for_codegen);

        // Dump IR if --disassemble is used.
        if (state_.disassemble) {
            util::log("IR for {:x}\n", pc);
            x86::backend::Dot_printer{}.run(graph_for_codegen);
            util::log("Translating {:x} to {:x}\n", pc, reinterpret_cast<uintptr_t>(block_ptr->code.data()));
        }

        // Lowering and target-specific lowering.
        ir::pass::Lowering{state_}.run(graph_for_codegen);
        ir::pass::Local_value_numbering{}.run(graph_for_codegen);
        x86::backend::Lowering{}.run(graph_for_codegen);

        // This garbage collection is required for Value::references to correctly reflect number of users.
        graph_for_codegen.garbage_collect();

        x86::Backend{state_, block_ptr->code}.run(graph_for_codegen);
        generate_eh_frame(*block_ptr);
    }

    // Update tag to reflect newly compiled code.
    icache_[tag] = block_ptr->code.data();
    icache_tag_[tag] = pc;
}
