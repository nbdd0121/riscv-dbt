#include "ir/pass.h"

namespace ir::pass {

bool Block_marker::before(Instruction* inst) {
    switch (inst->opcode()) {
        case Opcode::block:
            ASSERT(block_end);
            inst->attribute_pointer(block_end);
            block_end = nullptr;
            break;
        case Opcode::i_if:
        case Opcode::jmp:
            ASSERT(!block_end);
            block_end = inst;
            break;
        default:
            break;
    }
    return false;
}

}
