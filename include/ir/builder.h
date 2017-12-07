#ifndef IR_BUILDER_H
#define IR_BUILDER_H

#include "ir/instruction.h"

namespace ir {

class Builder {
private:
    Graph& _graph;
public:
    Builder(Graph& graph): _graph{graph} {}

    Instruction* constant(Type type, uint64_t value) {
        auto inst = new Instruction(type, Opcode::constant, {});
        inst->attribute(value);
        return _graph.manage(inst);
    }

    Instruction* cast(Type type, bool sext, Instruction* operand) {
        auto inst = new Instruction(type, Opcode::cast, {operand});
        inst->attribute(sext);
        return _graph.manage(inst);
    }

    Instruction* i_return(Instruction* dep) {
        return _graph.manage(new Instruction(Type::none, Opcode::i_return, {dep}));
    }

    Instruction* load_register(Instruction* dep, int regnum) {
        auto inst = new Instruction(Type::i64, Opcode::load_register, {dep});
        inst->attribute(regnum);
        return _graph.manage(inst);
    }

    Instruction* store_register(Instruction* dep, int regnum, Instruction* operand) {
        auto inst = new Instruction(Type::i64, Opcode::store_register, {dep, operand});
        inst->attribute(regnum);
        return _graph.manage(inst);
    }

    Instruction* load_memory(Instruction* dep, Type type, Instruction* address) {
        return _graph.manage(new Instruction(type, Opcode::load_memory, {dep, address}));
    }

    Instruction* store_memory(Instruction* dep, Instruction* address, Instruction* value) {
        return _graph.manage(new Instruction(Type::none, Opcode::store_memory, {dep, address, value}));
    }

    Instruction* emulate(Instruction* dep, void* ptr) {
        auto inst = new Instruction(Type::none, Opcode::emulate, {dep});
        inst->attribute_pointer(ptr);
        return _graph.manage(inst);
    }

    Instruction* arithmetic(Opcode opcode, Instruction* left, Instruction* right) {
        ASSERT(left->type() == right->type());
        return _graph.manage(new Instruction(left->type(), opcode, {left, right}));
    }

    Instruction* shift(Opcode opcode, Instruction* left, Instruction* right) {
        ASSERT(right->type() == Type::i8);
        return _graph.manage(new Instruction(left->type(), opcode, {left, right}));
    }

    Instruction* compare(Opcode opcode, Instruction* left, Instruction* right) {
        ASSERT(left->type() == right->type());
        return _graph.manage(new Instruction(Type::i1, opcode, {left, right}));
    }
};

}

#endif
