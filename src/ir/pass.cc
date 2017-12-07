#include "ir/instruction.h"
#include "ir/pass.h"

namespace ir::pass {

void Pass::run_recurse(Instruction* inst) {
    if (inst->_visited) return;
    if (before(inst)) {
        inst->_visited = 1;
        return;
    }

    inst->_visited = 2;

    // Visit all dependencies
    for (auto operand: inst->operands()) if (operand) run_recurse(operand);
    after(inst);
    inst->_visited = 1;
}

void Pass::run(Graph& graph) {
    for (auto inst: graph._heap) {
        inst->_visited = false;
    }
    run_recurse(graph.root());
}

}
