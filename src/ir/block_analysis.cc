#include <deque>
#include <list>

#include "ir/analysis.h"
#include "util/reverse_iterable.h"

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

void Block::enumerate_blocks() {
    std::deque<Node*> queue { *_graph.entry()->value(0).references().begin() };
    while (!queue.empty()) {
        auto node = queue.front();
        queue.pop_front();

        // Already visited.
        if (std::find(_blocks.begin(), _blocks.end(), node) != _blocks.end()) continue;

        _blocks.push_back(node);
        auto end = static_cast<Paired*>(node)->mate();
        for (auto value: end->values()) {
            for (auto ref: value.references()) {
                if (ref->opcode() == Opcode::exit) continue;
                queue.push_back(ref);
            }
        }
    }
}

void Block::update_keepalive() {
    std::deque<Node*> queue;

    bool trim_existing_keepalive = false;
    for (auto operand: _graph.exit()->operands()) {

        // Skip keepalive edges.
        if (operand.references().size() == 2) {
            trim_existing_keepalive = true;
            continue;
        }

        ASSERT(operand.opcode() != Opcode::entry);
        queue.push_back(static_cast<Paired*>(operand.node())->mate());
    }

    // Remove existing keepalive edges if any.
    if (trim_existing_keepalive) {
        auto operands = _graph.exit()->operands();
        // Remove all keepalive edges.
        operands.erase(
            std::remove_if(operands.begin(), operands.end(), [](auto operand) {
                return operand.references().size() == 2;
            }),
            operands.end()
        );
        _graph.exit()->operands(std::move(operands));
    }

    // Create a clone of the list of all blocks. Use list here for better erase performance.
    std::list<Node*> unseen_blocks(_blocks.begin(), _blocks.end());

    while (true) {
        while (!queue.empty()) {
            auto node = queue.front();
            queue.pop_front();

            auto ptr = std::find(unseen_blocks.begin(), unseen_blocks.end(), node);

            // Already visited.
            if (ptr == unseen_blocks.end()) continue;

            // Remove from unseen blocks
            unseen_blocks.erase(ptr);

            for (auto operand: node->operands()) {
                if (operand.opcode() == Opcode::entry) continue;
                queue.push_back(static_cast<Paired*>(operand.node())->mate());
            }
        }

        // All nodes have been visited.
        if (unseen_blocks.empty()) {
            break;
        }

        // Keepalive edges need to be inserted. Note that as a heuristic, we prefer blocks later in unseen blocks.
        for (auto block: util::reverse_iterable(unseen_blocks)) {

            // Only insert keepalive edges with jmp node.
            auto end = static_cast<Paired*>(block)->mate();
            if (end->opcode() == Opcode::jmp) {
                _graph.exit()->operand_add(end->value(0));
                queue.push_back(block);
                break;
            }
        }

        ASSERT(!queue.empty());
    }
}

}
