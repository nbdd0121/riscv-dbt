#ifndef X86_BACKEND_H
#define X86_BACKEND_H

#include "ir/pass.h"
#include "x86/encoder.h"
#include "x86/instruction.h"

#include <unordered_map>

namespace emu {
struct State;
}

namespace x86::backend {

namespace Target_opcode {

enum: uint16_t {
    // (Value base, Value index, Value scale, Value displacement) -> Value
    address = ir::Opcode::target_start,
    lea,
};

}

class Lowering: public ir::pass::Pass {
private:
    ir::Value match_address(ir::Value value, bool required);
protected:
    virtual void after(ir::Node* node) override;
};

class Dot_printer: public ir::pass::Dot_printer {
protected:
    virtual void write_node_content(std::ostream& stream, ir::Node* node) override;
};

}

namespace x86 {

class Backend: public ir::pass::Pass {
private:
    emu::State& _state;
    x86::Encoder _encoder;

    int stack_size = 0;
    std::unordered_map<ir::Value, int> reference_count;

    // Location of an node.
    std::unordered_map<ir::Value, Operand> location;

    // Spilled location of an node. This is used to avoid re-loading into memory if spilled again.
    std::unordered_map<ir::Value, Memory> memory_location;

    // Tracks what is stored in each register. Note that not all registers are used, but for easiness we still make its
    // size 16.
    std::array<ir::Value, 16> register_content {};

    // Tracks whether a register can be spilled, i.e. not pinned.
    std::array<bool, 16> pinned {};

public:
    Backend(emu::State& state, util::Code_buffer& buffer): _state {state}, _encoder{buffer} {}

    void emit(const Instruction& inst);
    void emit_move(ir::Type type, const Operand& dst, const Operand& src);

    /* Internal methods handling register allocation and spilling. */

    // Allocate a register without spilling. This is the fundamental operation for register allocation.
    Register alloc_register_no_spill(ir::Type type, Register hint);

    // Allocate a register, possibly spill other registers to memory. The hint will be respected only if it is a
    // register. The size of the hint will be ignored.
    Register alloc_register(ir::Type type, Register hint = Register::none);
    Register alloc_register(ir::Type type, const Operand& hint);

    // Spill a specified register to memory. Size of the register will be ignored.
    void spill_register(Register reg);

    // Spill all volatile registers to memory.
    void spill_all_registers();

    // Pin and unpin registers so they cannot be spilled. Size of the register will be ignored.
    void pin_register(Register reg);
    void unpin_register(Register reg);

    // Move a result to another location.
    void move_location(ir::Value value, const Operand& loc);

    // Bind a register to a node, and set up reference count.
    void bind_register(ir::Value value, Register reg);
    void ensure_register(ir::Value value, Register reg);
    void decrease_reference(ir::Value value);

    Operand get_location(ir::Value value);
    Operand get_location_ex(ir::Value value, bool allow_mem, bool allow_imm);

    // Get location, but guaranteed to be a register. This call might cause register to spill.
    Register get_register_location(ir::Value value);

    void emit_alu(ir::Node* node, Opcode opcode);
    void emit_shift(ir::Node* node, Opcode opcode);
    void emit_unary(ir::Node* node, Opcode opcode);
    void emit_mul(ir::Node* node, Opcode opcode);
    Condition_code emit_compare(ir::Value value);
    Memory emit_address(ir::Value value);

    void clear();

protected:
    virtual bool before(ir::Node* node) override { return node->opcode() == ir::Opcode::block; }
    virtual void after(ir::Node* node) override;

public:
    void run(ir::Graph& graph);
};

}

#endif
