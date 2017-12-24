#include <algorithm>

#include "ir/instruction.h"
#include "ir/pass.h"

namespace ir {

Instruction::Instruction(Type type, Opcode opcode, std::vector<Instruction*>&& dependencies, std::vector<Instruction*>&& operands):
    _dependencies(std::move(dependencies)), _operands(std::move(operands)), _type {type}, _opcode{opcode} {

    link();
}

Instruction::Instruction(const Instruction& inst):
    _dependencies(inst._dependencies), _operands(inst._operands),
    _attribute {inst._attribute}, _type {inst._type}, _opcode{inst._opcode} {

    link();
}

Instruction::Instruction(Instruction&& inst):
    _dependencies(std::move(inst._dependencies)), _operands(std::move(inst._operands)),
    _attribute {inst._attribute}, _type {inst._type}, _opcode{inst._opcode} {

    relink(&inst);
}

Instruction::~Instruction() {
    ASSERT(_references.size() == 0);
    unlink();
}

void Instruction::operator =(const Instruction& inst) {

    // Unlink will not create dangling reference here.
    unlink();

    // Copy _operands but not _references, as they are technically not part of the instruction.
    _dependencies = inst._dependencies;
    _operands = inst._operands;
    link();

    // Copy fields
    _attribute = inst._attribute;
    _type = inst._type;
    _opcode = inst._opcode;
}

void Instruction::operator =(Instruction&& inst) {

    // Unlink will not create dangling reference here.
    unlink();

    // Move _operands but not _references, as they are technically not part of the instruction.
    _dependencies = std::move(inst._dependencies);
    _operands = std::move(inst._operands);
    relink(&inst);

    // Copy fields
    _attribute = inst._attribute;
    _type = inst._type;
    _opcode = inst._opcode;
}

void Instruction::dependency_link() {
    for (auto dependency: _dependencies) {
        dependency->_dependants.insert(this);
    }
}

void Instruction::dependency_unlink() {
    for (auto dependency: _dependencies) {
        dependency->_dependants.remove(this);
    }
}

void Instruction::operand_link() {
    for (auto operand: _operands) {
        operand->_references.insert(this);
    }
}

void Instruction::operand_unlink() {
    for (auto operand: _operands) {
        operand->_references.remove(this);
    }
}

void Instruction::relink(Instruction* inst) {
    for (auto operand: _operands) {
        operand->_references.replace(inst, this);
    }
    for (auto dependency: _dependencies) {
        dependency->_dependants.replace(inst, this);
    }
}

void Instruction::dependencies(std::vector<Instruction*>&& dependencies) {
    dependency_unlink();
    _dependencies = std::move(dependencies);
    dependency_link();
}

void Instruction::dependency_update(Instruction* oldinst, Instruction* newinst) {
    ASSERT(oldinst && newinst);

    auto ptr = std::find(_dependencies.begin(), _dependencies.end(), oldinst);
    ASSERT(ptr != _dependencies.end());
    *ptr = newinst;
    newinst->_dependants.insert(this);
    oldinst->_dependants.remove(this);
}

void Instruction::dependency_add(Instruction* inst) {
    ASSERT(inst);
    _dependencies.push_back(inst);
    inst->_dependants.insert(this);
}

void Instruction::operands(std::vector<Instruction*>&& operands) {
    operand_unlink();
    _operands = std::move(operands);
    operand_link();
}

void Instruction::operand_set(size_t index, Instruction* inst) {
    ASSERT(index < _operands.size());
    ASSERT(inst);

    auto& ptr = _operands[index];
    inst->_references.insert(this);
    ptr->_references.remove(this);
    ptr = inst;
}

void Instruction::operand_update(Instruction* oldinst, Instruction* newinst) {
    ASSERT(oldinst && newinst);

    auto ptr = std::find(_operands.begin(), _operands.end(), oldinst);
    ASSERT(ptr != _operands.end());
    *ptr = newinst;
    newinst->_references.insert(this);
    oldinst->_references.remove(this);
}

Graph::Graph() {
    _start = manage(new Instruction(Type::none, Opcode::start, {}, {}));
}

Graph& Graph::operator=(Graph&& graph) {
    _heap.swap(graph._heap);
    _start = graph._start;
    _root = graph._root;
    return *this;
}

Graph::~Graph() {
    for (auto inst: _heap) {
        inst->_dependencies.clear();
        inst->_operands.clear();
        inst->_dependants.clear();
        inst->_references.clear();
        delete inst;
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
