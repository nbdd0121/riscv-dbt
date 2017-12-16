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
    size_t dependency_count = 0;
    switch (inst->opcode()) {
        case Opcode::end:
        case Opcode::block:
            control_dependency = true;
            dependency_count = inst->operand_count();
            break;
        case Opcode::if_true:
        case Opcode::if_false:
            control_dependency = true;
            dependency_count = 1;
            break;
        case Opcode::i_if:
        case Opcode::jmp:
        case Opcode::emulate:
        case Opcode::load_memory:
        case Opcode::store_memory:
            dependency_count = 1;
            break;
        case Opcode::fence:
            dependency_count = inst->operand_count();
            break;
        case Opcode::constant:
            std::clog << ' ' << static_cast<int64_t>(inst->attribute());
            break;
        case Opcode::cast:
            if (inst->attribute()) std::clog << " sext";
            break;
        case Opcode::load_register:
        case Opcode::store_register:
            dependency_count = 1;
            std::clog << " r" << inst->attribute();
            break;
        default: break;
    }
    std::clog << "\"];\n";

    // Draw operand dependencies as edges. For side-effect dependencies, we additionally color it blue.
    for (size_t i = 0; i < inst->operand_count(); i++) {
        auto operand = inst->operand(i);
        util::log(
            "\t\"{:x}\" -> \"{:x}\" [label={}{}];\n",
            reinterpret_cast<uintptr_t>(inst), reinterpret_cast<uintptr_t>(operand),
            i, i < dependency_count ? (control_dependency ? ",color=red" : ",color=blue") : ""
        );
    }
}

}
