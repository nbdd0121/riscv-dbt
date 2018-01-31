#include "ir/analysis.h"
#include "ir/pass.h"

#include "emu/state.h"
#include "util/reverse_iterable.h"

namespace ir::analysis {

void Load_store_elimination::populate_memops() {
    // Get lists of all memory operations within blocks.
    for (auto block: _block_analysis.blocks()) {
        _oplist = &_memops[block];
        visit_memops(static_cast<Paired*>(block)->mate());
        _visited.clear();
    }

    // Clean up as _visited is no longer needed.
    _visited = std::unordered_set<Node*>();
}

void Load_store_elimination::visit_memops(Node* node) {

    // Already visited.
    if (_visited.find(node) != _visited.end()) return;

    // Marked as visited.
    _visited.insert(node);

    // Visit memory predecessors first. This is a reverse post-order visit.
    for (auto op: node->operands()) {
        if (op.type() == Type::memory) {
            visit_memops(op.node());
        }
    }

    switch (node->opcode()) {
        case Opcode::load_register:
        case Opcode::store_register:
        case Opcode::load_memory:
        case Opcode::store_memory:
        case Opcode::call:
            _oplist->push_back(node);
            break;
    }
}

// First renaming pass. Fill operands of PHI nodes but no not touch the graph.
void Load_store_elimination::fill_store_phi(Node* block) {

    // Populate value stack.
    for (uint16_t regnum = 0; regnum < 66; regnum++) {

        // Rename PHI nodes first.
        auto phi_node = _phis[regnum].find(block);
        if (phi_node != _phis[regnum].end()) {
            _value_stack[regnum].push_back(phi_node->second->value(0));
        }
    }

    for (auto item: util::reverse_iterable(_memops[block])) {
        if (item->opcode() == Opcode::load_register) {
            uint16_t regnum = static_cast<Register_access*>(item)->regnum();
            _value_stack[regnum].push_back({});

        } else if (item->opcode() == Opcode::store_register) {
            uint16_t regnum = static_cast<Register_access*>(item)->regnum();
            _value_stack[regnum].push_back(item->value(0));

        } else if (emu::strict_exception ||
                    (item->opcode() == Opcode::call && static_cast<Call*>(item)->need_context())) {

            for (uint16_t regnum = 0; regnum < 66; regnum++) {
                _value_stack[regnum].push_back({});
            }
        }
    }

    // Fill in phi nodes operands.
    for (auto value: block->operands()) {
        if (value.opcode() == Opcode::entry) continue;

        auto node = static_cast<Paired*>(value.node())->mate();
        for (uint16_t regnum = 0; regnum < 66; regnum++) {
            auto phi_node = _phis[regnum].find(node);
            if (phi_node != _phis[regnum].end()) {
                auto value_stack_top = _value_stack[regnum].back();
                if (value_stack_top) {
                    phi_node->second->operand_set(value.index(), value_stack_top);
                }
            }
        }
    }

    // Recursively rename other blocks in the post-dominator tree.
    for (auto pred: _block_analysis.blocks()) {
        if (_dom.immediate_postdominator(pred) == block) {
            fill_store_phi(pred);
        }
    }

    // Pop values out from value stack.
    for (uint16_t regnum = 0; regnum < 66; regnum++) {
        auto phi_node = _phis[regnum].find(block);
        if (phi_node != _phis[regnum].end()) {
            _value_stack[regnum].pop_back();
        }
    }

    for (auto item: util::reverse_iterable(_memops[block])) {
        if (item->opcode() == Opcode::load_register || item->opcode() == Opcode::store_register) {
            uint16_t regnum = static_cast<Register_access*>(item)->regnum();
            _value_stack[regnum].pop_back();

        } else if (emu::strict_exception ||
                    (item->opcode() == Opcode::call && static_cast<Call*>(item)->need_context())) {

            for (uint16_t regnum = 0; regnum < 66; regnum++) {
                _value_stack[regnum].pop_back();
            }
        }
    }
}

// Second renaming phase. This time we are going to actually remove redundant stores.
void Load_store_elimination::rename_store(Node* block) {

    // Populate value stack.
    for (uint16_t regnum = 0; regnum < 66; regnum++) {
        auto phi_node = _phis[regnum].find(block);
        if (phi_node != _phis[regnum].end()) {

            // PHI nodes that are not killed indicates there are stores in each path.
            _value_stack[regnum].push_back(
                phi_node->second->operand_count() != 0 ? phi_node->second->value(0) : Value{}
            );
        }
    }

    for (auto item: util::reverse_iterable(_memops[block])) {
        if (item->opcode() == Opcode::load_register) {
            uint16_t regnum = static_cast<Register_access*>(item)->regnum();
            _value_stack[regnum].push_back({});

        } else if (item->opcode() == Opcode::store_register) {
            uint16_t regnum = static_cast<Register_access*>(item)->regnum();

            if (_value_stack[regnum].back()) {
                ir::pass::Pass::replace(item->value(0), item->operand(0));
            }

            _value_stack[regnum].push_back(item->value(0));

        } else if (emu::strict_exception ||
                    (item->opcode() == Opcode::call && static_cast<Call*>(item)->need_context())) {

            for (uint16_t regnum = 0; regnum < 66; regnum++) {
                _value_stack[regnum].push_back({});
            }
        }
    }

    // Recursively rename other blocks in the post-dominator tree.
    for (auto pred: _block_analysis.blocks()) {
        if (_dom.immediate_postdominator(pred) == block) {
            rename_store(pred);
        }
    }

    // Pop values out from value stack.
    for (uint16_t regnum = 0; regnum < 66; regnum++) {
        auto phi_node = _phis[regnum].find(block);
        if (phi_node != _phis[regnum].end()) {
            _value_stack[regnum].pop_back();
        }
    }

    for (auto item: util::reverse_iterable(_memops[block])) {
        if (item->opcode() == Opcode::load_register || item->opcode() == Opcode::store_register) {
            uint16_t regnum = static_cast<Register_access*>(item)->regnum();
            _value_stack[regnum].pop_back();

        } else if (emu::strict_exception ||
                    (item->opcode() == Opcode::call && static_cast<Call*>(item)->need_context())) {

            for (uint16_t regnum = 0; regnum < 66; regnum++) {
                _value_stack[regnum].pop_back();
            }
        }
    }
}

void Load_store_elimination::eliminate_store() {

    // This, along with fill_store_phi and rename_store, will eliminate all stores whose side-effects are definitely
    // overriden by later stores. It essentially does what SSA construction does, but on the reversed graph instead.
    // When a load or an instruction that may use the stored value is encountered, it pushes an invalid value into
    // the value stack. A PHI node that contains an invalid value is also consider invalid. As a result, if a store is
    // followed by an invalid value, it means that there will be an use after the store. If it is valid (i.e. either
    // followed by a store, or a PHI nodes that merges many paths that will eventually lead to a store), then the store
    // can be safely eliminated.

    // Just a dummy node to use as placeholder.
    Node dummy {Opcode::entry, {Type::none}, {}};

    // Build a working list for PHI node insertion.
    std::vector<std::vector<Node*>> worklist(66);
    for (auto& pair: _memops) {
        auto block = pair.first;
        std::vector<uint8_t> should_add(66);

        for (auto node: pair.second) {
            if (node->opcode() == Opcode::load_register || node->opcode() == Opcode::store_register) {
                auto regnum = static_cast<Register_access*>(node)->regnum();
                should_add[regnum] = 1;

            } else if (emu::strict_exception ||
                        (node->opcode() == Opcode::call && static_cast<Call*>(node)->need_context())) {

                // If strict exception is enabled, or the call takes CPU context as argument, then we consider it
                // as an operation that can potentially clobber the registers. We assume CPU contexts cannot be
                // retrieved elsewhere in the called function.

                for (uint16_t regnum = 0; regnum < 66; regnum++) {
                    should_add[regnum] = 1;
                }
            }
        }

        for (uint16_t regnum = 0; regnum < 66; regnum++) {
            if (should_add[regnum]) worklist[regnum].push_back(block);
        }
    }

    // Create PHI nodes based on post-dominance frontier. They are used for track-keeping purposes only in this
    // pass. There usage will not escape this pass, so we will not let them to be managed by the graph.
    _phis.resize(66);
    for (uint16_t regnum = 0; regnum < 66; regnum++) {
        auto& phi_map = _phis[regnum];
        auto list = std::move(worklist[regnum]);
        while (!list.empty()) {
            auto block = list.back();
            list.pop_back();

            for (auto frontier: _dom.postdominance_frontier(block)) {
                auto& phi_node = phi_map[frontier];
                if (!phi_node) {
                    auto end = static_cast<Paired*>(frontier)->mate();

                    // Normal phi nodes have n + 1 operands, but we don't care here.
                    phi_node = new Node(
                        Opcode::phi, {Type::none}, std::vector<Value>(end->value_count(), dummy.value(0))
                    );
                    list.push_back(frontier);
                }
            }
        }
    }

    // First renaming pass, to fill the operands of PHI nodes.
    for (auto pred: _block_analysis.blocks()) {
        if (_dom.immediate_postdominator(pred)->opcode() == Opcode::exit) {
            fill_store_phi(pred);
        }
    }

    // Now all PHI nodes referencing (directly or indirectly) the dummy node (i.e. invalid) would be invalidated.
    while (!dummy.value(0).references().empty()) {
        auto to_kill = *dummy.value(0).references().rbegin();
        to_kill->operands({});
        ir::pass::Pass::replace(to_kill->value(0), dummy.value(0));
    }

    // Second renaming pass to actually remove the stores.
    for (auto pred: _block_analysis.blocks()) {
        if (_dom.immediate_postdominator(pred)->opcode() == Opcode::exit) {
            rename_store(pred);
        }
    }

    // Destory all temporary PHI nodes.
    for (uint16_t regnum = 0; regnum < 66; regnum++) {
        for (auto& pair: _phis[regnum]) {
            pair.second->operands({});
        }

        for (auto& pair: _phis[regnum]) {
            delete pair.second;
        }
    }

    _phis = std::vector<std::unordered_map<Node*, Node*>>();
}

}
