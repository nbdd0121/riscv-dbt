#include "ir/pass.h"

namespace ir::pass {

bool Block_marker::before(Node* node) {
    switch (node->opcode()) {
        case Opcode::block:
            ASSERT(block_end);
            node->attribute_pointer(block_end);
            block_end = nullptr;
            break;
        case Opcode::i_if:
        case Opcode::jmp:
            ASSERT(!block_end);
            block_end = node;
            break;
        default:
            break;
    }
    return false;
}

}
