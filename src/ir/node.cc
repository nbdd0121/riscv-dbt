#include <algorithm>

#include "ir/node.h"
#include "ir/pass.h"

namespace ir {

Node::Node(uint16_t opcode, std::vector<Type>&& type, std::vector<Value>&& operands):
    _operands(std::move(operands)),  _type{std::move(type)}, _opcode{opcode}, _visited{0} {

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

Graph Graph::clone() const {
    Graph ret;
    std::unordered_map<Node*, Node*> mapping;

    // First create objects, but leave operands dummy.
    for (auto node: _heap) {
        Node* result;
        switch (node->opcode()) {
            case Opcode::start:
                // This node is already managed.
                mapping[node] = ret.start();
                continue;
            case Opcode::constant:
                result = new Constant(node->_type[0], static_cast<Constant*>(node)->const_value());
                break;
            case Opcode::cast:
                result = new Cast(node->_type[0], static_cast<Cast*>(node)->sign_extend(), ret.start()->value(0));
                break;
            case Opcode::load_register:
            case Opcode::store_register:
                result = new Register_access(
                    static_cast<Register_access*>(node)->regnum(),
                    node->_opcode,
                    std::vector<Type>(node->_type),
                    {}
                );
                break;
            case Opcode::block:
                result = new Block({});
                break;
            case Opcode::call:
                result = new Call(
                    static_cast<Call*>(node)->target(),
                    static_cast<Call*>(node)->need_context(),
                    std::vector<Type>(node->_type),
                    {}
                );
                break;
            default:
                result = new Node(node->_opcode, std::vector<Type>(node->_type), {});
                break;
        }
        mapping[node] = ret.manage(result);
    }

    // Fill objects
    for (auto node: _heap) {
        size_t op_count = node->_operands.size();

        std::vector<Value> operands(op_count);
        for (size_t i = 0; i < op_count; i++) {
            Value oldvalue = node->operand(i);
            operands[i] = { mapping[oldvalue.node()], oldvalue.index() };
        }
        mapping[node]->operands(std::move(operands));
    }

    ret.root(mapping[_root]);

    return std::move(ret);
}

}
