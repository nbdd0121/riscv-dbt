#ifndef IR_BUILDER_H
#define IR_BUILDER_H

#include "ir/instruction.h"

namespace ir::builder {

[[maybe_unused]]
static Instruction* constant(Type type, uint64_t value) {
    auto inst = new Instruction(type, Opcode::constant, {});
    inst->attribute(value);
    return inst;
}

[[maybe_unused]]
static Instruction* cast(Type type, bool sext, Instruction* operand) {
    auto inst = new Instruction(type, Opcode::cast, {operand});
    inst->attribute(sext);
    return inst;
}

[[maybe_unused]]
static Instruction* i_return(Instruction* dep) {
    auto inst = new Instruction(Type::none, Opcode::i_return, {dep});
    return inst;
}

[[maybe_unused]]
static Instruction* load_register(Instruction* dep, int regnum) {
    auto inst = new Instruction(Type::i64, Opcode::load_register, {dep});
    inst->attribute(regnum);
    return inst;
}

[[maybe_unused]]
static Instruction* store_register(Instruction* dep, int regnum, Instruction* operand) {
    auto inst = new Instruction(Type::i64, Opcode::store_register, {dep, operand});
    inst->attribute(regnum);
    return inst;
}

[[maybe_unused]]
static Instruction* load_memory(Instruction* dep, Type type, Instruction* address) {
    return new Instruction(type, Opcode::load_memory, {dep, address});
}

[[maybe_unused]]
static Instruction* store_memory(Instruction* dep, Instruction* address, Instruction* value) {
    return new Instruction(Type::none, Opcode::store_memory, {dep, address, value});
}

[[maybe_unused]]
static Instruction* emulate(Instruction* dep, void* ptr) {
    auto inst = new Instruction(Type::none, Opcode::emulate, {dep});
    inst->attribute_pointer(ptr);
    return inst;
}

[[maybe_unused]]
static Instruction* arithmetic(Opcode opcode, Instruction* left, Instruction* right) {
    ASSERT(left->type() == right->type());
    return new Instruction(left->type(), opcode, {left, right});
}

[[maybe_unused]]
static Instruction* shift(Opcode opcode, Instruction* left, Instruction* right) {
    ASSERT(right->type() == Type::i8);
    return new Instruction(left->type(), opcode, {left, right});
}

[[maybe_unused]]
static Instruction* compare(Opcode opcode, Instruction* left, Instruction* right) {
    ASSERT(left->type() == right->type());
    return new Instruction(Type::i1, opcode, {left, right});
}

}

#endif