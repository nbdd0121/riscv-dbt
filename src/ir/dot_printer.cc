#include "ir/instruction.h"
#include "ir/pass.h"

#include "util/assert.h"
#include "util/format.h"

namespace ir::pass {

void Dot_printer::start() {
    _index = 0;
    std::clog << "digraph G {\n\trankdir = BT;\n\tnode [shape=box];\n";
}

void Dot_printer::finish() {
    std::clog << "}" << std::endl;
}

bool Dot_printer::before(Instruction* inst) {

    // Allocate and assign index to the node. In contrast to Printer, we have to assign those returning none as well
    // here, as we are also displaying control and side-effect dependencies in the graph.
    // This has to be assigned before visiting children as the IR graph can be directed.
    inst->scratchpad(_index++);
    return false;
}

void Dot_printer::after(Instruction* inst) {

    // Draw the node with type, opcode
    std::clog << "\t" << inst->scratchpad() << " [label=\"";
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
            "\t{} -> {} [label={}{}];\n",
            inst->scratchpad(), operand->scratchpad(),
            i, i < dependency_count ? (control_dependency ? ",color=red" : ",color=blue") : ""
        );
    }
}

}
