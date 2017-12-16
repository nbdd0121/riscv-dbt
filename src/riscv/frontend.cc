#include "ir/builder.h"
#include "ir/instruction.h"
#include "riscv/basic_block.h"
#include "riscv/frontend.h"
#include "riscv/instruction.h"
#include "riscv/opcode.h"
#include "util/memory.h"

namespace riscv {

struct Frontend {
    ir::Graph graph;
    ir::Builder builder {graph};
    const Basic_block* block;

    // The last instruction with side-effect.
    ir::Instruction* last_side_effect = nullptr;

    ir::Instruction* emit_load_register(ir::Type type, int reg);
    void emit_store_register(int reg, ir::Instruction* value, bool sext = false);

    void emit_load(Instruction inst, ir::Type type, bool sext);
    void emit_store(Instruction inst, ir::Type type);
    void emit_alui(Instruction inst, ir::Opcode op, bool w);
    void emit_shifti(Instruction inst, ir::Opcode op, bool w);
    void emit_slti(Instruction inst, ir::Opcode op);
    void emit_alu(Instruction inst, ir::Opcode op, bool w);
    void emit_shift(Instruction inst, ir::Opcode op, bool w);
    void emit_slt(Instruction inst, ir::Opcode op);
    void emit_branch(Instruction instead, ir::Opcode op);

    void compile(const Basic_block& block);
};

ir::Instruction* Frontend::emit_load_register(ir::Type type, int reg) {
    ir::Instruction* ret;
    if (reg == 0) {
        ret = builder.constant(type, 0);
    } else {
        ret = builder.load_register(last_side_effect, reg);
        last_side_effect = ret;
        if (type != ir::Type::i64) ret = builder.cast(type, false, ret);
    }
    return ret;
}

void Frontend::emit_store_register(int reg, ir::Instruction* value, bool sext) {
    ASSERT(reg != 0);
    if (value->type() != ir::Type::i64) value = builder.cast(ir::Type::i64, sext, value);
    last_side_effect = builder.store_register(last_side_effect, reg, value);
}

void Frontend::emit_load(Instruction inst, ir::Type type, bool sext) {
    auto rs1_node = emit_load_register(ir::Type::i64, inst.rs1());
    auto imm_node = builder.constant(ir::Type::i64, inst.imm());
    auto address = builder.arithmetic(ir::Opcode::add, rs1_node, imm_node);
    auto rd_node = builder.load_memory(last_side_effect, type, address);
    last_side_effect = rd_node;
    emit_store_register(inst.rd(), rd_node, sext);
}

void Frontend::emit_store(Instruction inst, ir::Type type) {
    auto rs2_node = emit_load_register(type, inst.rs2());
    auto rs1_node = emit_load_register(ir::Type::i64, inst.rs1());
    auto imm_node = builder.constant(ir::Type::i64, inst.imm());
    auto address = builder.arithmetic(ir::Opcode::add, rs1_node, imm_node);
    last_side_effect = builder.store_memory(last_side_effect, address, rs2_node);
}

void Frontend::emit_alui(Instruction inst, ir::Opcode opcode, bool w) {
    if (inst.rd() == 0) return;
    ir::Type type = w ? ir::Type::i32 : ir::Type::i64;
    auto rs1_node = emit_load_register(type, inst.rs1());
    auto imm_node = builder.constant(type, inst.imm());
    auto rd_node = builder.arithmetic(opcode, rs1_node, imm_node);
    emit_store_register(inst.rd(), rd_node, true);
}

void Frontend::emit_shifti(Instruction inst, ir::Opcode opcode, bool w) {
    if (inst.rd() == 0) return;
    ir::Type type = w ? ir::Type::i32 : ir::Type::i64;
    auto rs1_node = emit_load_register(type, inst.rs1());
    auto imm_node = builder.constant(ir::Type::i8, inst.imm());
    auto rd_node = builder.shift(opcode, rs1_node, imm_node);
    emit_store_register(inst.rd(), rd_node, true);
}

void Frontend::emit_slti(Instruction inst, ir::Opcode opcode) {
    if (inst.rd() == 0) return;
    auto rs1_node = emit_load_register(ir::Type::i64, inst.rs1());
    auto imm_node = builder.constant(ir::Type::i64, inst.imm());
    auto rd_node = builder.compare(opcode, rs1_node, imm_node);
    emit_store_register(inst.rd(), rd_node);
}

void Frontend::emit_alu(Instruction inst, ir::Opcode opcode, bool w) {
    if (inst.rd() == 0) return;
    ir::Type type = w ? ir::Type::i32 : ir::Type::i64;
    auto rs1_node = emit_load_register(type, inst.rs1());
    auto rs2_node = emit_load_register(type, inst.rs2());
    auto rd_node = builder.arithmetic(opcode, rs1_node, rs2_node);
    emit_store_register(inst.rd(), rd_node, true);
}

void Frontend::emit_shift(Instruction inst, ir::Opcode opcode, bool w) {
    if (inst.rd() == 0) return;
    ir::Type type = w ? ir::Type::i32 : ir::Type::i64;
    auto rs1_node = emit_load_register(type, inst.rs1());
    auto rs2_node = emit_load_register(ir::Type::i8, inst.rs2());
    auto rd_node = builder.shift(opcode, rs1_node, rs2_node);
    emit_store_register(inst.rd(), rd_node, true);
}

void Frontend::emit_slt(Instruction inst, ir::Opcode opcode) {
    if (inst.rd() == 0) return;
    auto rs1_node = emit_load_register(ir::Type::i64, inst.rs1());
    auto rs2_node = emit_load_register(ir::Type::i64, inst.rs2());
    auto rd_node = builder.compare(opcode, rs1_node, rs2_node);
    emit_store_register(inst.rd(), rd_node);
}

void Frontend::emit_branch(Instruction inst, ir::Opcode opcode) {
    auto rs1_node = emit_load_register(ir::Type::i64, inst.rs1());
    auto rs2_node = emit_load_register(ir::Type::i64, inst.rs2());
    auto cmp_node = builder.compare(opcode, rs1_node, rs2_node);
    auto if_node = builder.control(ir::Opcode::i_if, {last_side_effect, cmp_node});
    auto if_true_node = builder.control(ir::Opcode::if_true, {if_node});
    auto if_false_node = builder.control(ir::Opcode::if_false, {if_node});

    // Building the true branch.
    auto true_block_node = builder.control(ir::Opcode::block, {if_true_node});
    auto pc_node = builder.load_register(true_block_node, 64);
    auto pc_offset = -inst.length() + inst.imm();
    auto pc_offset_node = builder.constant(ir::Type::i64, -inst.length() + inst.imm());
    auto new_pc_node = builder.arithmetic(ir::Opcode::add, pc_node, pc_offset_node);
    auto store_pc_node = builder.store_register(pc_node, 64, new_pc_node);
    auto true_jmp_node = builder.control(ir::Opcode::jmp, {store_pc_node});

    // If the jump target happens to be the basic block itself, create a loop.
    if (-pc_offset == block->end_pc - block->start_pc) {
        graph.start()->reference(0)->operand_add(true_jmp_node);
        auto end_node = builder.control(ir::Opcode::end, {if_false_node});
        graph.root(end_node);
        return;
    }

    auto end_node = builder.control(ir::Opcode::end, {true_jmp_node, if_false_node});
    graph.root(end_node);
}

void Frontend::compile(const Basic_block& block) {
    this->block = &block;

    auto start_node = graph.start();
    auto block_node = builder.control(ir::Opcode::block, {start_node});
    last_side_effect = block_node;

    // Update pc
    auto pc_node = builder.load_register(last_side_effect, 64);
    auto pc_offset_node = builder.constant(ir::Type::i64, block.end_pc - block.start_pc);
    auto new_pc_node = builder.arithmetic(ir::Opcode::add, pc_node, pc_offset_node);
    last_side_effect = builder.store_register(pc_node, 64, new_pc_node);

    // Update instret
    auto instret_node = builder.load_register(last_side_effect, 65);
    auto instret_offset_node = builder.constant(ir::Type::i64, block.instructions.size());
    auto new_instret_node = builder.arithmetic(ir::Opcode::add, instret_node, instret_offset_node);
    last_side_effect = builder.store_register(instret_node, 65, new_instret_node);

    riscv::reg_t pc_offset = block.start_pc - block.end_pc;
    for (auto& inst: block.instructions) {
        switch (inst.opcode()) {
            case Opcode::auipc: {
                if (inst.rd() == 0) break;
                auto pc_node = builder.load_register(last_side_effect, 64);
                auto offset_node = builder.constant(ir::Type::i64, pc_offset + inst.imm());
                auto rd_node = builder.arithmetic(ir::Opcode::add, pc_node, offset_node);
                last_side_effect = builder.store_register(pc_node, inst.rd(), rd_node);
                break;
            }
            case Opcode::lui: {
                if (inst.rd() == 0) break;
                auto imm_node = builder.constant(ir::Type::i64, inst.imm());
                last_side_effect = builder.store_register(last_side_effect, inst.rd(), imm_node);
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
                auto pc_node = builder.load_register(last_side_effect, 64);
                last_side_effect = pc_node;
                if (inst.rd()) {
                    last_side_effect = builder.store_register(last_side_effect, inst.rd(), pc_node);
                }
                ASSERT(pc_offset + inst.length() == 0);
                auto pc_offset_node = builder.constant(ir::Type::i64, -inst.length() + inst.imm());
                auto new_pc_node = builder.arithmetic(ir::Opcode::add, pc_node, pc_offset_node);
                last_side_effect = builder.store_register(last_side_effect, 64, new_pc_node);
                break;
            }
            case Opcode::jalr: {
                auto rs_node = emit_load_register(ir::Type::i64, inst.rs1());
                auto imm_node = builder.constant(ir::Type::i64, inst.imm());
                auto new_pc_node = builder.arithmetic(
                    ir::Opcode::i_and,
                    builder.arithmetic(ir::Opcode::add, rs_node, imm_node),
                    builder.constant(ir::Type::i64, ~1)
                );
                if (inst.rd()) {
                    auto pc_node = builder.load_register(last_side_effect, 64);
                    last_side_effect = builder.store_register(pc_node, inst.rd(), pc_node);
                }
                last_side_effect = builder.store_register(last_side_effect, 64, new_pc_node);
                break;
            }
            case Opcode::beq: emit_branch(inst, ir::Opcode::eq); return;
            case Opcode::bne: emit_branch(inst, ir::Opcode::ne); return;
            case Opcode::blt: emit_branch(inst, ir::Opcode::lt); return;
            case Opcode::bge: emit_branch(inst, ir::Opcode::ge); return;
            case Opcode::bltu: emit_branch(inst, ir::Opcode::ltu); return;
            case Opcode::bgeu: emit_branch(inst, ir::Opcode::geu); return;
            default: {
                last_side_effect = graph.manage(new ir::Instruction(ir::Type::none, ir::Opcode::emulate, {last_side_effect}));
                last_side_effect->attribute(util::read_as<uint64_t>(&inst));
                break;
            }
        }
        pc_offset += inst.length();
    }

    auto jmp_node = builder.control(ir::Opcode::jmp, {last_side_effect});
    auto end_node = builder.control(ir::Opcode::end, {jmp_node});
    graph.root(end_node);
}

ir::Graph compile(const Basic_block& block) {
    Frontend compiler;
    compiler.compile(block);
    return std::move(compiler.graph);
}

}
