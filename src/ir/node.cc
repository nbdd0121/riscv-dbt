#include <algorithm>

#include "ir/node.h"
#include "ir/pass.h"

namespace ir {

Node::Node(Opcode opcode, std::vector<Type>&& type, std::vector<Value>&& operands):
    _operands(std::move(operands)),  _type{std::move(type)}, _opcode{opcode} {

    link();
    _references.resize(_type.size());
}

Node::~Node() {
    for (auto ref: _references) ASSERT(ref.size() == 0);
    unlink();
}

void Node::link() {
    for (auto operand: _operands) {
        operand.node()->_references[operand.index()].insert(this);
    }
}

void Node::unlink() {
    for (auto operand: _operands) {
        operand.node()->_references[operand.index()].remove(this);
    }
}

void Node::operands(std::vector<Value>&& operands) {
    unlink();
    _operands = std::move(operands);
    link();
}

void Node::operand_set(size_t index, Value value) {
    ASSERT(index < _operands.size());

    auto& ptr = _operands[index];
    value.node()->_references[value.index()].insert(this);
    ptr.node()->_references[ptr.index()].remove(this);
    ptr = value;
}

void Node::operand_add(Value value) {
    _operands.push_back(value);
    value.node()->_references[value.index()].insert(this);
}

void Node::operand_update(Value oldvalue, Value newvalue) {
    auto ptr = std::find(_operands.begin(), _operands.end(), oldvalue);
    ASSERT(ptr != _operands.end());
    operand_set(ptr - _operands.begin(), newvalue);
}

Graph::Graph() {
    _start = manage(new Node(Opcode::start, {Type::control}, {}));
}

Graph& Graph::operator=(Graph&& graph) {
    _heap.swap(graph._heap);
    _start = graph._start;
    _root = graph._root;
    return *this;
}

Graph::~Graph() {
    for (auto node: _heap) {
        node->_operands.clear();
        node->_references.clear();
        delete node;
    }
}

void Graph::garbage_collect() {

    // Mark all reachable nodes.
    pass::Pass{}.run(*this);

    // Unlink to clear up references. This is necessary to maintain correctness of outgoing references.
    size_t size = _heap.size();
    for (size_t i = 0; i < size; i++) {
        if (!_heap[i]->_visited) {
            _heap[i]->unlink();
            _heap[i]->_operands.clear();
        }
    }

    for (size_t i = 0; i < size; i++) {
        if (!_heap[i]->_visited) {

            // Reclaim memory.
            delete _heap[i];

            // Move last element to current.
            _heap[i--] = _heap[--size];
        }
    }

    _heap.resize(size);
}

}
