#ifndef IR_INSTRUCTION_H
#define IR_INSTRUCTION_H

#include <cstdint>
#include <utility>
#include <vector>

#include "util/assert.h"

namespace ir {

namespace pass {
class Pass;
}

enum class Type: uint8_t {
    none = 0,
    i1 = 1,
    i8 = 8,
    i16 = 16,
    i32 = 32,
    i64 = 64,
};

enum class Opcode: uint8_t {
    /* Special instruction */
    constant,
    cast,
    i_return,
    emulate,

    /* Machine register load/store */
    load_register,
    store_register,

    /* Memory load/store */
    load_memory,
    store_memory,

    /* Binary ops */
    /* Arithmetic operations */
    add,
    sub,
    i_xor,
    i_or,
    i_and,

    /* Shift operations */
    shl,
    shr,
    sar,

    /* Compare */
    eq,
    ne,
    lt,
    ge,
    ltu,
    geu,

    /* Unary ops */
    neg,
    i_not,
};

[[maybe_unused]]
static bool is_binary_opcode(Opcode opcode) {
    uint8_t value = static_cast<uint8_t>(opcode);
    return value >= static_cast<uint8_t>(Opcode::add) && value <= static_cast<uint8_t>(Opcode::geu);
}

class Instruction {
private:

    // Instructions that this instruction references.
    std::vector<Instruction*> _operands;

    // Instructions that references this instruction.
    std::vector<Instruction*> _references;

    // Additional attributes for some instructions.
    union {
        uint64_t value;
        void *pointer;
    } _attribute;

    // Scratchpad for passes to store data temporarily.
    union {
        uint64_t value;
        void *pointer;
    } _scratchpad;

    // The output type of this instruction.
    Type _type;

    // Opcode of the instruction.
    Opcode _opcode;

    // Whether the instruction is visited. For graph walking only.
    bool _visited;

public:
    Instruction(Type type, Opcode opcode, std::vector<Instruction*>&& operands);
    Instruction(const Instruction& inst);
    Instruction(Instruction&& inst);
    ~Instruction();
    
    void operator =(const Instruction& inst);
    void operator =(Instruction&& inst);

private:
    // Controls whether this instruction should be added as themselves as referencing its operands.
    void link();
    void unlink();
    void relink(Instruction* inst);

public:
    // Field accessors and mutators
    uint64_t scratchpad() const { return _scratchpad.value; }
    void scratchpad(uint64_t value) { _scratchpad.value = value; }
    void* scratchpad_pointer() const { return _scratchpad.pointer; }
    void scratchpad_pointer(void* pointer) { _scratchpad.pointer = pointer; }

    uint64_t attribute() const { return _attribute.value; }
    void attribute(uint64_t value) { _attribute.value = value; }
    void* attribute_pointer() const { return _attribute.pointer; }
    void attribute_pointer(void* pointer) { _attribute.pointer = pointer; }

    Type type() const { return _type; }
    void type(Type type) { _type = type; }
    Opcode opcode() const { return _opcode; }
    void opcode(Opcode opcode) { _opcode = opcode; }

    // Operand accessors and mutators
    const std::vector<Instruction*>& operands() const { return _operands; }
    void operands(std::vector<Instruction*>&& operands) { _operands = std::move(operands); }
    size_t operand_count() const { return _operands.size(); }

    Instruction* operand(size_t index) const { 
        ASSERT(index < _operands.size());
        return _operands[index];
    }

    void operand_set(size_t index, Instruction* inst);
    void operand_swap(size_t first, size_t second) { std::swap(_operands[first], _operands[second]); }
    void operand_update(Instruction* oldinst, Instruction* newinst);

    // Reference accessors and mutators
    const std::vector<Instruction*>& references() const { return _references; }
    size_t reference_count() { return _references.size(); }
    Instruction* reference(size_t index) const {
        ASSERT(index < _references.size());
        return _references[index];
    }
    
    void reference_add(Instruction* inst) { _references.push_back(inst); }
    void reference_remove(Instruction* inst);
    void reference_update(Instruction* oldinst, Instruction* newinst);
    
    friend class Instruction_heap;
    friend pass::Pass;
};

class Instruction_heap {
public:
    std::vector<Instruction*> _heap;

public:
    ~Instruction_heap();

    Instruction* manage(Instruction* inst) {
        _heap.push_back(inst);
        return inst;
    }
};

} // ir

#endif
