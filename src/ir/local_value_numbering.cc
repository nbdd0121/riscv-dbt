#include "ir/node.h"
#include "ir/pass.h"

namespace ir::pass {

size_t Local_value_numbering::Hash::operator ()(Value value) const noexcept {
    Node *node = value.node();
    size_t hash = value.index() ^ static_cast<uint8_t>(node->opcode());

    ASSERT(is_pure_opcode(node->opcode()));

    for (size_t i = 0; i < node->value_count(); i++) {
        hash ^= static_cast<uint8_t>(node->value(i).type());
    }

    for (auto operand: node->operands()) {
        hash ^= reinterpret_cast<uintptr_t>(operand.node()) ^ operand.index();
    }

    switch (node->opcode()) {
        case Opcode::constant:
        case Opcode::cast:
            hash ^= node->attribute();
            break;
        default: break;
    }

    return hash;
}

bool Local_value_numbering::Equal_to::operator ()(Value a, Value b) const noexcept {
    Node *anode = a.node();
    Node *bnode = b.node();

    if (a.index() != b.index()) return false;

    if (anode->opcode() != bnode->opcode()) return false;

    ASSERT(is_pure_opcode(anode->opcode()));

    if (anode->value_count() != bnode->value_count()) return false;

    for (size_t i = 0; i < anode->value_count(); i++) {
        if (anode->value(i).type() != bnode->value(i).type()) return false;
    }

    size_t operand_count = anode->operand_count();
    if (operand_count != bnode->operand_count()) return false;

    for (size_t i = 0; i < operand_count; i++) {
        if (anode->operand(i) != bnode->operand(i)) return false;
    }

    switch (anode->opcode()) {
        case Opcode::constant:
        case Opcode::cast:
            if (anode->attribute() != bnode->attribute()) return false;
            break;
        default: break;
    }

    return true;
}

// Helper function that replaces current value with a constant value. It will keep type intact.
void Local_value_numbering::replace_with_constant(Value value, uint64_t const_value) {

    // Create a new constant node.
    Node *const_node = _graph->manage(new Node(Opcode::constant, {value.type()}, {}));
    const_node->attribute(const_value);
    Value new_value = const_node->value(0);

    auto pair = _set.insert(new_value);
    replace(value, pair.second ? new_value : *pair.first);
}

void Local_value_numbering::lvn(Value value) {
    // perform the actual local value numbering.
    // Try insert into the set. If insertion succeeded, then this is a new value, so return.
    auto pair = _set.insert(value);
    if (pair.second) return;

    // Otherwise replace with the existing one.
    if (value != *pair.first) replace(value, *pair.first);
}

void Local_value_numbering::after(Node* node) {
    auto opcode = node->opcode();

    if (!is_pure_opcode(opcode)) return;

    if (opcode == Opcode::constant) {
        return lvn(node->value(0));
    }

    if (opcode == Opcode::cast) {
        // Folding cast node.
        auto output = node->value(0);
        auto x = node->operand(0);

        // If the operand is constant, then perform constant folding.
        if (x.is_const()) {
            return replace_with_constant(
                output,
                Evaluator::cast(output.type(), x.type(), node->attribute(), x.const_value())
            );
        }

        // Two casts can be possibly folded.
        if (x.opcode() == Opcode::cast) {
            auto y = x.node()->operand(0);
            size_t ysize = get_type_size(y.type());
            size_t size = get_type_size(output.type());

            // If the size is same, then eliminate.
            if (ysize == size) {
                return replace(output, y);
            }

            // A down-cast followed by an up-cast cannot be folded.
            size_t xsize = get_type_size(x.type());
            if (ysize > xsize && xsize < size) return lvn(output);

            // An up-cast followed by an up-cast cannot be folded if sext does not match.
            if (ysize < xsize && xsize < size && x.node()->attribute() != node->attribute()) return lvn(output);

            // This can either be up-cast followed by up-cast, up-cast followed by down-cast.
            // As the result is an up-cast, we need to select the correct sext.
            if (ysize < size) {
                node->attribute(x.node()->attribute());
            }

            node->operand_set(0, y);
        }

        return lvn(output);
    }

    if (is_binary_opcode(opcode)) {
        // Folding binary operation node.
        auto output = node->value(0);
        auto x = node->operand(0);
        auto y = node->operand(1);

        // If both operands are constant, then perform constant folding.
        if (x.is_const() && y.is_const()) {
            return replace_with_constant(
                output,
                Evaluator::binary(x.type(), node->opcode(), x.const_value(), y.const_value())
            );
        }

        // Canonicalization, for commutative opcodes move constant to the right.
        // TODO: For non-abelian comparisions, we can also move constant to the right by performing transformations on
        // immediate.
        if (x.is_const()) {
            if (is_commutative_opcode(opcode)) {
                node->operand_swap(0, 1);
                std::swap(x, y);
            } else {
                if (x.const_value() == 0) {
                    // Arithmetic identity folding for non-abelian operations.
                    switch (opcode) {
                        case Opcode::sub:
                            node->opcode(Opcode::neg);
                            node->operands({y});
                            return lvn(output);
                        case Opcode::shl:
                        case Opcode::shr:
                        case Opcode::sar:
                            return replace_with_constant(output, 0);
                        // 0 < unsigned is identical to unsigned != 0
                        case Opcode::ltu:
                            node->opcode(Opcode::ne);
                            node->operand_swap(0, 1);
                            return lvn(output);
                        // 0 >= unsigned is identical to unsigned == 0
                        case Opcode::geu:
                            node->opcode(Opcode::eq);
                            node->operand_swap(0, 1);
                            return lvn(output);
                        default: break;
                    }
                }
            }
        }

        // Arithmetic identity folding.
        // TODO: Other arithmetic identity worth considering:
        // x + x == x << 1
        // x >> 63 == x < 0
        if (y.is_const()) {
            if (y.const_value() == 0) {
                switch (opcode) {
                    // For these node x @ 0 == x
                    case Opcode::add:
                    case Opcode::sub:
                    case Opcode::i_xor:
                    case Opcode::i_or:
                    case Opcode::shl:
                    case Opcode::shr:
                    case Opcode::sar:
                        return replace(output, x);
                    // For these node x @ 0 == 0
                    case Opcode::i_and:
                    case Opcode::ltu:
                        return replace_with_constant(output, 0);
                    // unsigned >= 0 is tautology
                    case Opcode::geu:
                        return replace_with_constant(output, 1);
                    default: break;
                }
            } else if (y.const_value() == static_cast<uint64_t>(-1)) {
                switch (opcode) {
                    case Opcode::i_xor:
                        node->opcode(Opcode::i_not);
                        node->operands({x});
                        return lvn(output);
                    case Opcode::i_and:
                        return replace(output, x);
                    case Opcode::i_or:
                        return replace_with_constant(output, -1);
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
                    return replace_with_constant(output, 0);
                case Opcode::i_or:
                case Opcode::i_and:
                    return replace(output, x);
                case Opcode::eq:
                case Opcode::ge:
                case Opcode::geu:
                    return replace_with_constant(output, 1);
                default: break;
            }
        }

        // More folding for add
        if (opcode == Opcode::add && y.references().size() == 1) {
            if (x.opcode() == Opcode::add && x.node()->operand(1).is_const()) {
                y.node()->attribute(
                    Evaluator::sign_extend(output.type(), y.const_value() + x.node()->operand(1).const_value())
                );
                x = x.node()->operand(0);
                node->operand_set(0, x);
            }
        }

        return lvn(output);
    }

    if (opcode == Opcode::mux) {
        auto output = node->value(0);
        auto x = node->operand(0);
        auto y = node->operand(1);
        auto z = node->operand(2);

        if (x.is_const()) {
            return replace(output, x.const_value() ? y : z);
        }

        return lvn(output);
    }

    ASSERT(0);
}

}
