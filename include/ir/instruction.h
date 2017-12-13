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

[[maybe_unused]]
static size_t get_type_size(Type type) {
    return static_cast<uint8_t>(type);
}

enum class Opcode: uint8_t {
    /** Control flow opcodes **/
    start,
    i_return,

    /** Opcodes with side-effects **/
    emulate,
    fence,

    /* Machine register load/store */
    load_register,
    store_register,

    /* Memory load/store */
    load_memory,
    store_memory,

    /** Pure opcodes **/
    constant,
    cast,

    /* Unary ops */
    neg,
    i_not,

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
};

[[maybe_unused]]
static bool is_pure_opcode(Opcode opcode) {
    return static_cast<uint8_t>(opcode) >= static_cast<uint8_t>(Opcode::constant);
}

[[maybe_unused]]
static bool is_binary_opcode(Opcode opcode) {
    uint8_t value = static_cast<uint8_t>(opcode);
    return value >= static_cast<uint8_t>(Opcode::add) && value <= static_cast<uint8_t>(Opcode::geu);
}

[[maybe_unused]]
static bool is_commutative_opcode(Opcode opcode) {
    switch(opcode) {
        case Opcode::add:
        case Opcode::i_xor:
        case Opcode::i_or:
        case Opcode::i_and:
        case Opcode::eq:
        case Opcode::ne:
            return true;
        default:
            return false;
    }
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
    // 0 - not visited, 1 - visited, 2 - visiting.
    uint8_t _visited;

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
    
    friend class Graph;
    friend pass::Pass;
};

class Graph {
private:
    std::vector<Instruction*> _heap;
    Instruction* _root = nullptr;

public:
    Graph() {}
    Graph(const Graph&) = delete;
    Graph(Graph&&) = default;
    ~Graph();

    Graph& operator =(const Graph&) = delete;
    Graph& operator =(Graph&&) = default;

    Instruction* manage(Instruction* inst) {
        _heap.push_back(inst);
        return inst;
    }

    // Free up dead instructions. Not necessary during compilation, but useful for reducing footprint when graph needs
    // to be cached.
    void garbage_collect();

    Instruction* root() const { return _root; }
    void root(Instruction* root) { _root = root; }

    friend pass::Pass;
};

} // ir

#endif
