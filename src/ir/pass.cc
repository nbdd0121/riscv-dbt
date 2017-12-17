#include "ir/instruction.h"
#include "ir/pass.h"

namespace ir::pass {

void Pass::replace(Instruction* oldnode, Instruction* newnode) {
    while (!oldnode->dependants().empty()) {
        (*oldnode->dependants().rbegin())->dependency_update(oldnode, newnode);
    }

    while (!oldnode->references().empty()) {
        (*oldnode->references().rbegin())->operand_update(oldnode, newnode);
    }
}

void Pass::run_recurse(Instruction* inst) {
    if (inst->_visited) return;
    if (before(inst)) {
        inst->_visited = 1;
        return;
    }

    inst->_visited = 2;

    // Visit all dependencies
    for (auto dependency: inst->dependencies()) run_recurse(dependency);
    for (auto operand: inst->operands()) run_recurse(operand);
    after(inst);
    inst->_visited = 1;
}

void Pass::run_on(Graph& graph, Instruction* inst) {
    _graph = &graph;
    for (auto inst: graph._heap) {
        inst->_visited = false;
    }
    start();
    run_recurse(inst);
    finish();
}

}
