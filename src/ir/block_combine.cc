#include "ir/pass.h"

namespace ir::pass {

void Block_combine::after(Node* node) {

    // If the block has only one entry point, and it is from a jump, then merge.
    if (node->opcode() == Opcode::block && node->operand_count() == 1 && node->operand(0).opcode() == Opcode::jmp) {

        // Retrieve the start and end node of both blocks.
        auto next_block = static_cast<ir::Paired*>(node);
        auto next_jmp = static_cast<ir::Paired*>(next_block->mate());
        auto prev_jmp = static_cast<ir::Paired*>(node->operand(0).node());
        auto prev_block = static_cast<ir::Paired*>(prev_jmp->mate());

        // Link two blocks together.
        replace(node->value(0), prev_jmp->operand(0));

        // Update mate information.
        next_jmp->mate(prev_block);
        prev_block->mate(next_jmp);
    }
}

}
