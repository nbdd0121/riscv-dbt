#ifndef IR_BUILDER_H
#define IR_BUILDER_H

#include "ir/node.h"

namespace ir {

class Builder {
private:
    Graph& _graph;
public:
    Builder(Graph& graph): _graph{graph} {}

    Node* create(Type type, Opcode opcode, std::vector<Node*>&& dep, std::vector<Node*>&& opr) {
        return _graph.manage(new Node(type, opcode, std::move(dep), std::move(opr)));
    }

    Node* control(Opcode opcode, std::vector<Node*>&& dep) {
        return create(Type::none, opcode, std::move(dep), {});
    }

    Node* constant(Type type, uint64_t value) {
        auto inst = create(type, Opcode::constant, {}, {});
        inst->attribute(value);
        return inst;
    }

    Node* cast(Type type, bool sext, Node* operand) {
        auto inst = create(type, Opcode::cast, {}, {operand});
        inst->attribute(sext);
        return inst;
    }

    Node* load_register(Node* dep, int regnum) {
        auto inst = create(Type::i64, Opcode::load_register, {dep}, {});
        inst->attribute(regnum);
        return inst;
    }

    Node* store_register(Node* dep, int regnum, Node* operand) {
        auto inst = create(Type::none, Opcode::store_register, {dep}, {operand});
        inst->attribute(regnum);
        return inst;
    }

    Node* load_memory(Node* dep, Type type, Node* address) {
        return create(type, Opcode::load_memory, {dep}, {address});
    }

    Node* store_memory(Node* dep, Node* address, Node* value) {
        return create(Type::none, Opcode::store_memory, {dep}, {address, value});
    }

    Node* arithmetic(Opcode opcode, Node* left, Node* right) {
        ASSERT(left->type() == right->type());
        return create(left->type(), opcode, {}, {left, right});
    }

    Node* shift(Opcode opcode, Node* left, Node* right) {
        ASSERT(right->type() == Type::i8);
        return create(left->type(), opcode, {}, {left, right});
    }

    Node* compare(Opcode opcode, Node* left, Node* right) {
        ASSERT(left->type() == right->type());
        return create(Type::i1, opcode, {}, {left, right});
    }

    Node* mux(Node* cond, Node* left, Node* right) {
        ASSERT(cond->type() == Type::i1 && left->type() == right->type());
        return create(left->type(), Opcode::mux, {}, {cond, left, right});
    }
};

}

#endif
