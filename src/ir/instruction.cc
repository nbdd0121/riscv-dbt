#include <algorithm>

#include "ir/instruction.h"

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
        if (operand) operand->reference_add(this);
    }
}

void Instruction::unlink() {
    for (auto operand: _operands) {
        if (operand) operand->reference_remove(this);
    }
}

void Instruction::relink(Instruction* inst) {
    for (auto operand: _operands) {
        if (operand) operand->reference_update(inst, this);
    }
    
}

void Instruction::operand_set(size_t index, Instruction* inst) {
    ASSERT(index < _operands.size());
    auto& ptr = _operands[index];
    if (inst) inst->reference_add(inst);
    if (ptr) ptr->reference_remove(inst);
    ptr = inst;
}

void Instruction::operand_update(Instruction* oldinst, Instruction* newinst) {
    auto ptr = std::find(_operands.begin(), _operands.end(), oldinst);
    ASSERT(ptr != _operands.end());
    *ptr = newinst;
    newinst->reference_add(this);
    oldinst->reference_remove(this);
}

void Instruction::reference_remove(Instruction* inst) {
    auto ptr = std::find(_references.begin(), _references.end(), inst);
    ASSERT(ptr != _references.end());

    // Swap the last element to the current place
    auto last_element = _references.end() - 1;
    if (ptr != last_element) {
        *ptr = *last_element;
    }

    _references.pop_back();
}

void Instruction::reference_update(Instruction* oldinst, Instruction* newinst) {
    auto ptr = std::find(_references.begin(), _references.end(), oldinst);
    ASSERT(ptr != _references.end());
    *ptr = newinst;
}

Instruction_heap::~Instruction_heap() {
    for (auto inst: _heap) {
        inst->_operands.clear();
        inst->_references.clear();
        delete inst;
    }
}

}
