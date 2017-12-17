#include "ir/instruction.h"
#include "ir/pass.h"

#include "util/assert.h"
#include "util/format.h"

namespace ir::pass {

const char* Dot_printer::opcode_name(Opcode opcode) {
    switch (opcode) {
#define CASE(name) case Opcode::name: return #name;
        CASE(start)
        CASE(end)
        CASE(block)
        case Opcode::i_if: return "if";
        CASE(if_true)
        CASE(if_false)
        CASE(jmp)
        CASE(constant)
        CASE(cast)
        CASE(fence)
        CASE(load_register)
        CASE(store_register)
        CASE(load_memory)
        CASE(store_memory)
        CASE(emulate)
        CASE(add)
        CASE(sub)
        case Opcode::i_xor: return "xor";
        case Opcode::i_or: return "or";
        case Opcode::i_and: return "and";
        CASE(shl)
        CASE(shr)
        CASE(sar)
        CASE(eq)
        CASE(ne)
        CASE(lt)
        CASE(ge)
        CASE(ltu)
        CASE(geu)
        CASE(neg)
        case Opcode::i_not: return "not";
#undef CASE
        default: return "(unknown)";
    }
}

const char* Dot_printer::type_name(Type type) {
    switch (type) {
#define CASE(name) case Type::name: return #name;
        CASE(none)
        CASE(i1)
        CASE(i8)
        CASE(i16)
        CASE(i32)
        CASE(i64)
#undef CASE
        default: return "(unknown)";
    }
}

void Dot_printer::start() {
    std::clog << "digraph G {\n\trankdir = BT;\n\tnode [shape=box];\n";
}

void Dot_printer::finish() {
    std::clog << "}" << std::endl;
}

void Dot_printer::after(Instruction* inst) {

    // Draw the node with type, opcode
    util::log("\t\"{:x}\" [label=\"", reinterpret_cast<uintptr_t>(inst));
    if (inst->type() != Type::none) {
        std::clog << type_name(inst->type()) << ' ';
    }
    std::clog << opcode_name(inst->opcode());

    // Also append attributes to node label, and decide number of control or side-effect dependencies operands.
    bool control_dependency = false;
    switch (inst->opcode()) {
        case Opcode::end:
        case Opcode::block:
        case Opcode::if_true:
        case Opcode::if_false:
            control_dependency = true;
            break;
        case Opcode::constant:
            std::clog << ' ' << static_cast<int64_t>(inst->attribute());
            break;
        case Opcode::cast:
            if (inst->attribute()) std::clog << " sext";
            break;
        case Opcode::load_register:
        case Opcode::store_register:
            std::clog << " r" << inst->attribute();
            break;
        default: break;
    }
    std::clog << "\"];\n";

    // Draw dependencies as edges. Data flow dependencies are colored black, control flow dependencies are colored red,
    // and side-effect dependnecies are colored blue.
    auto dependencies = inst->dependencies();
    for (size_t i = 0; i < dependencies.size(); i++) {
        auto operand = dependencies[i];
        util::log(
            "\t\"{:x}\" -> \"{:x}\" [label={}{}];\n",
            reinterpret_cast<uintptr_t>(inst), reinterpret_cast<uintptr_t>(operand),
            i, control_dependency ? ",color=red" : ",color=blue"
        );
    }

    auto operands = inst->operands();
    for (size_t i = 0; i < operands.size(); i++) {
        auto operand = operands[i];
        util::log(
            "\t\"{:x}\" -> \"{:x}\" [label={}{}];\n",
            reinterpret_cast<uintptr_t>(inst), reinterpret_cast<uintptr_t>(operand), i, ""
        );
    }
}

}
