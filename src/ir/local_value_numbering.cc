#include "ir/instruction.h"
#include "ir/pass.h"

namespace ir::pass {

size_t Local_value_numbering::Hash::operator ()(Instruction* inst) const noexcept {

    size_t hash = static_cast<uint8_t>(inst->opcode());
    hash ^= static_cast<uint8_t>(inst->type());

    for (auto operand: inst->operands()) {
        hash ^= reinterpret_cast<uintptr_t>(operand);
    }

    switch (inst->opcode()) {
        case Opcode::constant:
        case Opcode::cast:
        case Opcode::emulate:
        case Opcode::load_register:
        case Opcode::store_register:
            hash ^= inst->attribute();
            break;
        default: break;
    }

    return hash;
}

bool Local_value_numbering::Equal_to::operator ()(Instruction* a, Instruction* b) const noexcept {
    if (a->opcode() != b->opcode()) return false;
    if (a->type() != b->type()) return false;

    size_t operand_count = a->operand_count();
    if (operand_count != b->operand_count()) return false;

    for (size_t i = 0; i < operand_count; i++) {
        if (a->operand(i) != b->operand(i)) return false;
    }

    switch (a->opcode()) {
        case Opcode::constant:
        case Opcode::cast:
        case Opcode::emulate:
        case Opcode::load_register:
        case Opcode::store_register:
            if (a->attribute() != b->attribute()) return false;
            break;
        default: break;
    }

    return true;
}

// Helper function that replaces current instruction with a constant node. It will keep type intact.
void Local_value_numbering::replace_with_constant(Instruction* inst, uint64_t value) {
    inst->attribute(value);
    inst->opcode(Opcode::constant);
    inst->operands({});
}

void Local_value_numbering::after(Instruction* inst) {
    auto opcode = inst->opcode();

    if (!is_pure_opcode(opcode)) return;

    if (opcode == Opcode::cast) {
        // Folding cast node.
        auto x = inst->operand(0);

        // If the operand is constant, then perform constant folding.
        if (x->opcode() == Opcode::constant) {
            replace_with_constant(inst, Evaluator::cast(inst->type(), x->type(), inst->attribute(), x->attribute()));
            goto lvn;
        }

        // Two casts can be possibly folded.
        if (x->opcode() == Opcode::cast) {
            auto y = x->operand(0);
            size_t ysize = get_type_size(y->type());
            size_t size = get_type_size(inst->type());

            // If the size is same, then eliminate.
            if (ysize == size) {
                replace(inst, y);
                return;
            }

            // A down-cast followed by an up-cast cannot be folded.
            size_t xsize = get_type_size(x->type());
            if (ysize > xsize && xsize < size) goto lvn;

            // An up-cast followed by an up-cast cannot be folded if sext does not match.
            if (ysize < xsize && xsize < size && x->attribute() != inst->attribute()) goto lvn;

            // This can either be up-cast followed by up-cast, up-cast followed by down-cast.
            // As the result is an up-cast, we need to select the correct sext.
            if (ysize < size) {
                inst->attribute(x->attribute());
            }

            inst->operand_set(0, y);
            goto lvn;
        }

    } else if (is_binary_opcode(opcode)) {
        // Folding binary operation node.
        auto x = inst->operand(0);
        auto y = inst->operand(1);

        // If both operands are constant, then perform constant folding.
        if (x->opcode() == Opcode::constant && y->opcode() == Opcode::constant) {
            replace_with_constant(inst, Evaluator::binary(x->type(), inst->opcode(), x->attribute(), y->attribute()));
            goto lvn;
        }

        // Canonicalization, for commutative opcodes move constant to the right.
        // TODO: For non-abelian comparisions, we can also move constant to the right by performing transformations on
        // immediate.
        if (x->opcode() == Opcode::constant) {
            if (is_commutative_opcode(opcode)) {
                inst->operand_swap(0, 1);
                std::swap(x, y);
            } else {
                if (x->attribute() == 0) {
                    // Arithmetic identity folding for non-abelian operations.
                    switch (opcode) {
                        case Opcode::sub:
                            inst->opcode(Opcode::neg);
                            inst->operands({y});
                            goto lvn;
                        case Opcode::shl:
                        case Opcode::shr:
                        case Opcode::sar:
                            replace_with_constant(inst, 0);
                            goto lvn;
                        // 0 < unsigned is identical to unsigned != 0
                        case Opcode::ltu:
                            inst->opcode(Opcode::ne);
                            inst->operand_swap(0, 1);
                            goto lvn;
                        // 0 >= unsigned is identical to unsigned == 0
                        case Opcode::geu:
                            inst->opcode(Opcode::eq);
                            inst->operand_swap(0, 1);
                            goto lvn;
                        default: break;
                    }
                }
            }
        }

        // Arithmetic identity folding.
        // TODO: Other arithmetic identity worth considering:
        // x + x == x << 1
        // x >> 63 == x < 0
        if (y->opcode() == Opcode::constant) {
            if (y->attribute() == 0) {
                switch (opcode) {
                    // For these instruction x @ 0 == x
                    case Opcode::add:
                    case Opcode::sub:
                    case Opcode::i_xor:
                    case Opcode::i_or:
                    case Opcode::shl:
                    case Opcode::shr:
                    case Opcode::sar:
                        replace(inst, x);
                        return;
                    // For these instruction x @ 0 == 0
                    case Opcode::i_and:
                    case Opcode::ltu:
                        replace_with_constant(inst, 0);
                        goto lvn;
                    // unsigned >= 0 is tautology
                    case Opcode::geu:
                        replace_with_constant(inst, 1);
                        goto lvn;
                    default: break;
                }
            } else if (y->attribute() == static_cast<uint64_t>(-1)) {
                switch (opcode) {
                    case Opcode::i_xor:
                        inst->opcode(Opcode::i_not);
                        inst->operands({x});
                        goto lvn;
                    case Opcode::i_and:
                        replace(inst, x);
                        return;
                    case Opcode::i_or:
                        replace_with_constant(inst, -1);
                        goto lvn;
                    default: break;
                }
            }
        }

        if (x == y) {
            switch (opcode) {
                case Opcode::sub:
                case Opcode::i_xor:
                case Opcode::ne:
                case Opcode::lt:
                case Opcode::ltu:
                    replace_with_constant(inst, 0);
                    goto lvn;
                case Opcode::i_or:
                case Opcode::i_and:
                    replace(inst, x);
                    return;
                case Opcode::eq:
                case Opcode::ge:
                case Opcode::geu:
                    replace_with_constant(inst, 1);
                    goto lvn;
                default: break;
            }
        }

        // More folding for add
        if (opcode == Opcode::add && y->references().size() == 1) {
            if (x->opcode() == Opcode::add && x->operand(1)->opcode() == Opcode::constant) {
                y->attribute(Evaluator::sign_extend(inst->type(), y->attribute() + x->operand(1)->attribute()));
                x = x->operand(0);
                inst->operand_set(0, x);
                goto lvn;
            }
        }
    }

lvn:
    // Now perform the actual local value numbering.
    // Try insert into the set. If insertion succeeded, then this is a new node, so return.
    auto pair = _set.insert(inst);
    if (pair.second) return;

    // Otherwise replace with the existing one.
    replace(inst, *pair.first);
}

}
