#include "x86/backend.h"

namespace x86::backend {

void Dot_printer::write_node_content(std::ostream& stream, ir::Node* node) {
    if (!ir::is_target_specific(node->opcode())) {
        return ir::pass::Dot_printer::write_node_content(stream, node);
    }

    switch (node->opcode()) {
        case Target_opcode::address: stream << "x86::address"; break;
        case Target_opcode::lea: stream << "x86::lea"; break;
        default: ASSERT(0);
    }
}

}
