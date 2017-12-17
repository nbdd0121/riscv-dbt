#include <algorithm>

#include "ir/instruction.h"
#include "ir/pass.h"

namespace ir {

Instruction::Instruction(Type type, Opcode opcode, std::vector<Instruction*>&& operands):
    _operands(std::move(operands)), _type {type}, _opcode{opcode} {

    link();
}

Instruction::Instruction(const Instruction& inst):
    _operands(inst._operands), _attribute {inst._attribute}, _type {inst._type}, _opcode{inst._opcode} {

    link();
}

Instruction::Instruction(Instruction&& inst):
    _operands(std::move(inst._operands)), _attribute {inst._attribute}, _type {inst._type}, _opcode{inst._opcode} {

    relink(&inst);
}

Instruction::~Instruction() {
    ASSERT(_references.size() == 0);
    unlink();
}

void Instruction::operator =(const Instruction& inst) {

    // Unlike will not create dangling reference here.
    unlink();

    // Copy _operands but not _references, as they are technically not part of the instruction.
    _operands = inst._operands;
    link();

    // Copy fields
    _attribute = inst._attribute;
    _type = inst._type;
    _opcode = inst._opcode;
}

void Instruction::operator =(Instruction&& inst) {

    // Unlike will not create dangling reference here.
    unlink();

    // Move _operands but not _references, as they are technically not part of the instruction.
    _operands = std::move(inst._operands);
    relink(&inst);

    // Copy fields
    _attribute = inst._attribute;
    _type = inst._type;
    _opcode = inst._opcode;
}

void Instruction::link() {
    for (auto operand: _operands) {
        operand->_references.insert(this);
    }
}

void Instruction::unlink() {
    for (auto operand: _operands) {
        operand->_references.remove(this);
    }
}

void Instruction::relink(Instruction* inst) {
    for (auto operand: _operands) {
        operand->_references.replace(inst, this);
    }
}

void Instruction::operands(std::vector<Instruction*>&& operands) {
    unlink();
    _operands = std::move(operands);
    link();
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

void Instruction::operand_add(Instruction* inst) {
    ASSERT(inst);
    _operands.push_back(inst);
    inst->_references.insert(this);
}

Graph::Graph() {
    _start = manage(new Instruction(Type::none, Opcode::start, {}));
}

Graph::~Graph() {
    for (auto inst: _heap) {
        inst->_operands.clear();
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
