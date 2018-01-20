#include "ir/analysis.h"

namespace ir::analysis {

Node* Block::get_target(Value control) {
    size_t refcount = control.references().size();
    ASSERT(refcount == 1 || refcount == 2);
    bool skip_exit = refcount == 2;
    for (auto ref: control.references()) {
        if (skip_exit && ref->opcode() == Opcode::exit) continue;
        return ref;
    }
    ASSERT(0);
}

Value Block::get_tail_jmp_pc(Value control, uint16_t pc_regnum) {
    size_t refcount = control.references().size();
    if (refcount != 1) {
        // This jmp contains a keepalive edge, it therefore cannot be a tail jump.
        ASSERT(refcount == 2);
        return {};
    }

    auto target = *control.references().begin();

    // Not tail position
    if (target->opcode() != ir::Opcode::exit) return {};

    auto last_mem = control.node()->operand(0);
    if (last_mem.opcode() == ir::Opcode::fence) {
        for (auto operand: last_mem.node()->operands()) {
            if (operand.opcode() == ir::Opcode::store_register &&
                static_cast<ir::Register_access*>(operand.node())->regnum() == pc_regnum) {

                return operand.node()->operand(1);
            }
        }

    } else if (last_mem.opcode() == ir::Opcode::store_register &&
               static_cast<ir::Register_access*>(last_mem.node())->regnum() == pc_regnum) {

        return last_mem.node()->operand(1);

    }

    return {};
}

}
