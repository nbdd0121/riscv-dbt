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

void Dot_printer::after(Instruction* inst) {

    // Allocate and assign index to the node. In contrast to Printer, we have to assign those returning none as well
    // here, as we are also displaying side-effect dependencies in the graph.
    inst->scratchpad(_index++);

    // Draw the node with type, opcode
    std::clog << "\t" << inst->scratchpad() << " [label=\"";
    if (inst->type() != Type::none) {
        std::clog << type_name(inst->type()) << ' ';
    }
    std::clog << opcode_name(inst->opcode());

    // Also append attributes to node label, and decide number of side-effect dependencies operands.
    size_t dependency_count = 0;
    switch (inst->opcode()) {
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
        case Opcode::load_memory:
        case Opcode::store_memory:
        case Opcode::emulate:
        case Opcode::i_return:
            dependency_count = 1;
            break;
        default: break;
    }
    std::clog << "\"];\n";

    // Draw operand dependencies as edges. For side-effect dependencies, we additionally color it blue.
    for (size_t i = 0; i < inst->operand_count(); i++) {
        auto operand = inst->operand(i);
        if (operand) {
            util::log(
                "\t{} -> {} [label={}{}];\n",
                inst->scratchpad(), operand->scratchpad(),
                i, i < dependency_count ? ",color=blue" : ""
            );
        }
    }
}

}
