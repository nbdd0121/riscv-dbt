#include "emu/state.h"
#include "ir/builder.h"
#include "ir/node.h"
#include "riscv/basic_block.h"
#include "riscv/context.h"
#include "riscv/frontend.h"
#include "riscv/instruction.h"
#include "riscv/opcode.h"
#include "util/memory.h"

namespace riscv {

struct Frontend {
    ir::Graph graph;
    ir::Builder builder {graph};
    emu::State& state;
    const Basic_block* block;

    // The latest memory value.
    ir::Value last_memory;

    Frontend(emu::State& state): state{state} {}

    ir::Value emit_load_register(ir::Type type, uint16_t reg);
    void emit_store_register(uint16_t reg, ir::Value value, bool sext = false);

    void emit_load(Instruction inst, ir::Type type, bool sext);
    void emit_store(Instruction inst, ir::Type type);
    void emit_alui(Instruction inst, uint16_t opcode, bool w);
    void emit_shifti(Instruction inst, uint16_t opcode, bool w);
    void emit_slti(Instruction inst, uint16_t opcode);
    void emit_alu(Instruction inst, uint16_t opcode, bool w);
    void emit_shift(Instruction inst, uint16_t opcode, bool w);
    void emit_slt(Instruction inst, uint16_t opcode);
    void emit_branch(Instruction instead, uint16_t opcode, emu::reg_t pc);

    void compile(const Basic_block& block);
};

ir::Value Frontend::emit_load_register(ir::Type type, uint16_t reg) {
    ir::Value ret;
    if (reg == 0) {
        ret = builder.constant(type, 0);
    } else {
        std::tie(last_memory, ret) = builder.load_register(last_memory, reg);
        if (type != ir::Type::i64) ret = builder.cast(type, false, ret);
    }
    return ret;
}

void Frontend::emit_store_register(uint16_t reg, ir::Value value, bool sext) {
    ASSERT(reg != 0);
    if (value.type() != ir::Type::i64) value = builder.cast(ir::Type::i64, sext, value);
    last_memory = builder.store_register(last_memory, reg, value);
}

void Frontend::emit_load(Instruction inst, ir::Type type, bool sext) {
    auto rs1_value = emit_load_register(ir::Type::i64, inst.rs1());
    auto imm_value = builder.constant(ir::Type::i64, inst.imm());
    auto address = builder.arithmetic(ir::Opcode::add, rs1_value, imm_value);
    ir::Value rd_value;
    std::tie(last_memory, rd_value) = builder.load_memory(last_memory, type, address);
    emit_store_register(inst.rd(), rd_value, sext);
}

void Frontend::emit_store(Instruction inst, ir::Type type) {
    auto rs2_value = emit_load_register(type, inst.rs2());
    auto rs1_value = emit_load_register(ir::Type::i64, inst.rs1());
    auto imm_value = builder.constant(ir::Type::i64, inst.imm());
    auto address = builder.arithmetic(ir::Opcode::add, rs1_value, imm_value);
    last_memory = builder.store_memory(last_memory, address, rs2_value);
}

void Frontend::emit_alui(Instruction inst, uint16_t opcode, bool w) {
    if (inst.rd() == 0) return;
    ir::Type type = w ? ir::Type::i32 : ir::Type::i64;
    auto rs1_value = emit_load_register(type, inst.rs1());
    auto imm_value = builder.constant(type, inst.imm());
    auto rd_value = builder.arithmetic(opcode, rs1_value, imm_value);
    emit_store_register(inst.rd(), rd_value, true);
}

void Frontend::emit_shifti(Instruction inst, uint16_t opcode, bool w) {
    if (inst.rd() == 0) return;
    ir::Type type = w ? ir::Type::i32 : ir::Type::i64;
    auto rs1_value = emit_load_register(type, inst.rs1());
    auto imm_value = builder.constant(ir::Type::i8, inst.imm());
    auto rd_value = builder.shift(opcode, rs1_value, imm_value);
    emit_store_register(inst.rd(), rd_value, true);
}

void Frontend::emit_slti(Instruction inst, uint16_t opcode) {
    if (inst.rd() == 0) return;
    auto rs1_value = emit_load_register(ir::Type::i64, inst.rs1());
    auto imm_value = builder.constant(ir::Type::i64, inst.imm());
    auto rd_value = builder.compare(opcode, rs1_value, imm_value);
    emit_store_register(inst.rd(), rd_value);
}

void Frontend::emit_alu(Instruction inst, uint16_t opcode, bool w) {
    if (inst.rd() == 0) return;
    ir::Type type = w ? ir::Type::i32 : ir::Type::i64;
    auto rs1_value = emit_load_register(type, inst.rs1());
    auto rs2_value = emit_load_register(type, inst.rs2());
    auto rd_value = builder.arithmetic(opcode, rs1_value, rs2_value);
    emit_store_register(inst.rd(), rd_value, true);
}

void Frontend::emit_shift(Instruction inst, uint16_t opcode, bool w) {
    if (inst.rd() == 0) return;
    ir::Type type = w ? ir::Type::i32 : ir::Type::i64;
    auto rs1_value = emit_load_register(type, inst.rs1());
    auto rs2_value = emit_load_register(ir::Type::i8, inst.rs2());
    auto rd_value = builder.shift(opcode, rs1_value, rs2_value);
    emit_store_register(inst.rd(), rd_value, true);
}

void Frontend::emit_slt(Instruction inst, uint16_t opcode) {
    if (inst.rd() == 0) return;
    auto rs1_value = emit_load_register(ir::Type::i64, inst.rs1());
    auto rs2_value = emit_load_register(ir::Type::i64, inst.rs2());
    auto rd_value = builder.compare(opcode, rs1_value, rs2_value);
    emit_store_register(inst.rd(), rd_value);
}

void Frontend::emit_branch(Instruction inst, uint16_t opcode, emu::reg_t pc) {
    auto rs1_value = emit_load_register(ir::Type::i64, inst.rs1());
    auto rs2_value = emit_load_register(ir::Type::i64, inst.rs2());
    auto cmp_value = builder.compare(opcode, rs1_value, rs2_value);

    auto new_pc_value = builder.constant(ir::Type::i64, pc + inst.imm());

    bool use_mux = true;
    if (pc + inst.imm() == block->start_pc) use_mux = false;

    if (use_mux) {
        auto pc_value = builder.constant(ir::Type::i64, block->end_pc);
        auto mux_value = builder.mux(cmp_value, new_pc_value, pc_value);
        auto store_pc_value = builder.store_register(last_memory, 64, mux_value);
        auto jmp_value = builder.control(ir::Opcode::jmp, {store_pc_value});
        auto end_value = builder.control(ir::Opcode::end, {jmp_value});
        graph.root(end_value.node());

    } else {
        auto if_node = builder.create(ir::Opcode::i_if, {ir::Type::control, ir::Type::control}, {last_memory, cmp_value});

        // Building the true branch.
        auto true_block_value = builder.block({if_node->value(0)});
        auto store_pc_value = builder.store_register(true_block_value, 64, new_pc_value);
        auto true_jmp_value = builder.control(ir::Opcode::jmp, {store_pc_value});

        // If the jump target happens to be the basic block itself, create a loop.
        if (pc + inst.imm() == block->start_pc) {
            (*graph.start()->value(0).references().begin())->operand_add(true_jmp_value);
            auto end_value = builder.control(ir::Opcode::end, {if_node->value(1)});
            graph.root(end_value.node());
            return;
        }

        auto end_value = builder.control(ir::Opcode::end, {true_jmp_value, if_node->value(1)});
        graph.root(end_value.node());
    }
}

void Frontend::compile(const Basic_block& block) {
    this->block = &block;

    auto start_value = graph.start()->value(0);
    auto block_value = builder.block({start_value});
    last_memory = block_value;

    // Update pc
    auto end_pc_value = builder.constant(ir::Type::i64, block.end_pc);
    last_memory = builder.store_register(last_memory, 64, end_pc_value);

    // Update instret
    if (!state.no_instret) {
        ir::Value instret_value;
        std::tie(last_memory, instret_value) = builder.load_register(last_memory, 65);
        auto instret_offset_value = builder.constant(ir::Type::i64, block.instructions.size());
        auto new_instret_value = builder.arithmetic(ir::Opcode::add, instret_value, instret_offset_value);
        last_memory = builder.store_register(last_memory, 65, new_instret_value);
    }

    riscv::reg_t pc = block.start_pc;
    for (auto& inst: block.instructions) {
        switch (inst.opcode()) {
            case Opcode::auipc: {
                if (inst.rd() == 0) break;
                auto rd_value = builder.constant(ir::Type::i64, pc + inst.imm());
                last_memory = builder.store_register(last_memory, inst.rd(), rd_value);
                break;
            }
            case Opcode::lui: {
                if (inst.rd() == 0) break;
                auto imm_value = builder.constant(ir::Type::i64, inst.imm());
                last_memory = builder.store_register(last_memory, inst.rd(), imm_value);
                break;
            }
            case Opcode::lb: emit_load(inst, ir::Type::i8, true); break;
            case Opcode::lh: emit_load(inst, ir::Type::i16, true); break;
            case Opcode::lw: emit_load(inst, ir::Type::i32, true); break;
            case Opcode::ld: emit_load(inst, ir::Type::i64, false); break;
            case Opcode::lbu: emit_load(inst, ir::Type::i8, false); break;
            case Opcode::lhu: emit_load(inst, ir::Type::i16, false); break;
            case Opcode::lwu: emit_load(inst, ir::Type::i32, false); break;
            case Opcode::sb: emit_store(inst, ir::Type::i8); break;
            case Opcode::sh: emit_store(inst, ir::Type::i16); break;
            case Opcode::sw: emit_store(inst, ir::Type::i32); break;
            case Opcode::sd: emit_store(inst, ir::Type::i64); break;
            case Opcode::addi: emit_alui(inst, ir::Opcode::add, false); break;
            case Opcode::slli: emit_shifti(inst, ir::Opcode::shl, false); break;
            case Opcode::slti: emit_slti(inst, ir::Opcode::lt); break;
            case Opcode::sltiu: emit_slti(inst, ir::Opcode::ltu); break;
            case Opcode::xori: emit_alui(inst, ir::Opcode::i_xor, false); break;
            case Opcode::srli: emit_shifti(inst, ir::Opcode::shr, false); break;
            case Opcode::srai: emit_shifti(inst, ir::Opcode::sar, false); break;
            case Opcode::ori: emit_alui(inst, ir::Opcode::i_or, false); break;
            case Opcode::andi: emit_alui(inst, ir::Opcode::i_and, false); break;
            case Opcode::addiw: emit_alui(inst, ir::Opcode::add, true); break;
            case Opcode::slliw: emit_shifti(inst, ir::Opcode::shl, true); break;
            case Opcode::srliw: emit_shifti(inst, ir::Opcode::shr, true); break;
            case Opcode::sraiw: emit_shifti(inst, ir::Opcode::sar, true); break;
            case Opcode::add: emit_alu(inst, ir::Opcode::add, false); break;
            case Opcode::sub: emit_alu(inst, ir::Opcode::sub, false); break;
            case Opcode::sll: emit_shift(inst, ir::Opcode::shl, false); break;
            case Opcode::slt: emit_slt(inst, ir::Opcode::lt); break;
            case Opcode::sltu: emit_slt(inst, ir::Opcode::ltu); break;
            case Opcode::i_xor: emit_alu(inst, ir::Opcode::i_xor, false); break;
            case Opcode::srl: emit_shift(inst, ir::Opcode::shr, false); break;
            case Opcode::sra: emit_shift(inst, ir::Opcode::sar, false); break;
            case Opcode::i_or: emit_alu(inst, ir::Opcode::i_or, false); break;
            case Opcode::i_and: emit_alu(inst, ir::Opcode::i_and, false); break;
            case Opcode::addw: emit_alu(inst, ir::Opcode::add, true); break;
            case Opcode::subw: emit_alu(inst, ir::Opcode::sub, true); break;
            case Opcode::sllw: emit_shift(inst, ir::Opcode::shl, true); break;
            case Opcode::srlw: emit_shift(inst, ir::Opcode::shr, true); break;
            case Opcode::sraw: emit_shift(inst, ir::Opcode::sar, true); break;
            case Opcode::jal: {
                if (inst.rd()) {
                    last_memory = builder.store_register(last_memory, inst.rd(), end_pc_value);
                }
                ASSERT(pc + inst.length() == block.end_pc);
                auto new_pc_value = builder.constant(ir::Type::i64, pc + inst.imm());
                last_memory = builder.store_register(last_memory, 64, new_pc_value);
                break;
            }
            case Opcode::jalr: {
                auto rs_value = emit_load_register(ir::Type::i64, inst.rs1());
                auto imm_value = builder.constant(ir::Type::i64, inst.imm());
                auto new_pc_value = builder.arithmetic(
                    ir::Opcode::i_and,
                    builder.arithmetic(ir::Opcode::add, rs_value, imm_value),
                    builder.constant(ir::Type::i64, ~1)
                );
                if (inst.rd()) {
                    last_memory = builder.store_register(last_memory, inst.rd(), end_pc_value);
                }
                last_memory = builder.store_register(last_memory, 64, new_pc_value);
                break;
            }
            case Opcode::beq: emit_branch(inst, ir::Opcode::eq, pc); return;
            case Opcode::bne: emit_branch(inst, ir::Opcode::ne, pc); return;
            case Opcode::blt: emit_branch(inst, ir::Opcode::lt, pc); return;
            case Opcode::bge: emit_branch(inst, ir::Opcode::ge, pc); return;
            case Opcode::bltu: emit_branch(inst, ir::Opcode::ltu, pc); return;
            case Opcode::bgeu: emit_branch(inst, ir::Opcode::geu, pc); return;
            default: {
                auto serialized_inst = builder.constant(ir::Type::i64, util::read_as<uint64_t>(&inst));
                last_memory = graph.manage(new ir::Call(
                    reinterpret_cast<uintptr_t>(step), true, {ir::Type::memory}, {last_memory, serialized_inst}
                ))->value(0);
                break;
            }
        }
        pc += inst.length();
    }

    auto jmp_value = builder.control(ir::Opcode::jmp, {last_memory});
    auto end_value = builder.control(ir::Opcode::end, {jmp_value});
    graph.root(end_value.node());
}

ir::Graph compile(emu::State& state, const Basic_block& block) {
    Frontend compiler {state};
    compiler.compile(block);
    return std::move(compiler.graph);
}

}
