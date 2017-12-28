#ifndef IR_BUILDER_H
#define IR_BUILDER_H

#include <tuple>

#include "ir/node.h"

namespace ir {

class Builder {
private:
    Graph& _graph;
public:
    Builder(Graph& graph): _graph{graph} {}

    Node* create(Opcode opcode, std::vector<Type>&& type, std::vector<Value>&& opr) {
        return _graph.manage(new Node(opcode, std::move(type), std::move(opr)));
    }

    Value control(Opcode opcode, std::vector<Value>&& opr) {
        return create(opcode, {Type::control}, std::move(opr))->value(0);
    }

    Value constant(Type type, uint64_t value) {
        auto inst = create(Opcode::constant, {type}, {});
        inst->attribute(value);
        return inst->value(0);
    }

    Value cast(Type type, bool sext, Value operand) {
        auto inst = create(Opcode::cast, {type}, {operand});
        inst->attribute(sext);
        return inst->value(0);
    }

    std::tuple<Value, Value> load_register(Value dep, int regnum) {
        auto inst = create(Opcode::load_register, {Type::memory, Type::i64}, {dep});
        inst->attribute(regnum);
        return {inst->value(0), inst->value(1)};
    }

    Value store_register(Value dep, int regnum, Value operand) {
        auto inst = create(Opcode::store_register, {Type::memory}, {dep, operand});
        inst->attribute(regnum);
        return inst->value(0);
    }

    std::tuple<Value, Value> load_memory(Value dep, Type type, Value address) {
        auto inst = create(Opcode::load_memory, {Type::memory, type}, {dep, address});
        return {inst->value(0), inst->value(1)};
    }

    Value store_memory(Value dep, Value address, Value value) {
        return create(Opcode::store_memory, {Type::memory}, {dep, address, value})->value(0);
    }

    Value arithmetic(Opcode opcode, Value left, Value right) {
        ASSERT(left.type() == right.type());
        return create(opcode, {left.type()}, {left, right})->value(0);
    }

    Value shift(Opcode opcode, Value left, Value right) {
        ASSERT(right.type() == Type::i8);
        return create(opcode, {left.type()}, {left, right})->value(0);
    }

    Value compare(Opcode opcode, Value left, Value right) {
        ASSERT(left.type() == right.type());
        return create(opcode, {Type::i1}, {left, right})->value(0);
    }

    Value mux(Value cond, Value left, Value right) {
        ASSERT(cond.type() == Type::i1 && left.type() == right.type());
        return create(Opcode::mux, {left.type()}, {cond, left, right})->value(0);
    }
};

}

#endif
