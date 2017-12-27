#include <algorithm>

#include "ir/node.h"
#include "ir/pass.h"

namespace ir {

Node::Node(Type type, Opcode opcode, std::vector<Node*>&& dependencies, std::vector<Node*>&& operands):
    _dependencies(std::move(dependencies)), _operands(std::move(operands)), _type {type}, _opcode{opcode} {

    link();
}

Node::~Node() {
    ASSERT(_references.size() == 0);
    unlink();
}

void Node::dependency_link() {
    for (auto dependency: _dependencies) {
        dependency->_dependants.insert(this);
    }
}

void Node::dependency_unlink() {
    for (auto dependency: _dependencies) {
        dependency->_dependants.remove(this);
    }
}

void Node::operand_link() {
    for (auto operand: _operands) {
        operand->_references.insert(this);
    }
}

void Node::operand_unlink() {
    for (auto operand: _operands) {
        operand->_references.remove(this);
    }
}

void Node::dependencies(std::vector<Node*>&& dependencies) {
    dependency_unlink();
    _dependencies = std::move(dependencies);
    dependency_link();
}

void Node::dependency_update(Node* oldinst, Node* newinst) {
    ASSERT(oldinst && newinst);

    auto ptr = std::find(_dependencies.begin(), _dependencies.end(), oldinst);
    ASSERT(ptr != _dependencies.end());
    *ptr = newinst;
    newinst->_dependants.insert(this);
    oldinst->_dependants.remove(this);
}

void Node::dependency_add(Node* node) {
    ASSERT(node);
    _dependencies.push_back(node);
    node->_dependants.insert(this);
}

void Node::operands(std::vector<Node*>&& operands) {
    operand_unlink();
    _operands = std::move(operands);
    operand_link();
}

void Node::operand_set(size_t index, Node* node) {
    ASSERT(index < _operands.size());
    ASSERT(node);

    auto& ptr = _operands[index];
    node->_references.insert(this);
    ptr->_references.remove(this);
    ptr = node;
}

void Node::operand_update(Node* oldinst, Node* newinst) {
    ASSERT(oldinst && newinst);

    auto ptr = std::find(_operands.begin(), _operands.end(), oldinst);
    ASSERT(ptr != _operands.end());
    *ptr = newinst;
    newinst->_references.insert(this);
    oldinst->_references.remove(this);
}

Graph::Graph() {
    _start = manage(new Node(Type::none, Opcode::start, {}, {}));
}

Graph& Graph::operator=(Graph&& graph) {
    _heap.swap(graph._heap);
    _start = graph._start;
    _root = graph._root;
    return *this;
}

Graph::~Graph() {
    for (auto node: _heap) {
        node->_dependencies.clear();
        node->_operands.clear();
        node->_dependants.clear();
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
            _heap[i]->_dependencies.clear();
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
