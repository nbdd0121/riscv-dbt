#ifndef IR_VISIT_H
#define IR_VISIT_H

#include "ir/node.h"

namespace ir {

template<typename F>
void visit_local_memops_reverse_postorder(Node* node, F func) {

    // Memory nodes are chained as a list (not a DAG). Therefore we don't have to track whether a node is visited.
    for (auto op: node->operands()) {
        if (op.type() == Type::memory) {
            visit_local_memops_reverse_postorder(op.node(), func);
        }
    }

    switch (node->opcode()) {
        case Opcode::load_register:
        case Opcode::store_register:
        case Opcode::load_memory:
        case Opcode::store_memory:
        case Opcode::call:
            func(node);
            break;
    }
}

}

#endif
