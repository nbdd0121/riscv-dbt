#include <deque>
#include <functional>

#include "ir/analysis.h"

namespace ir {

void Dominance::compute_blocks() {
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

void Dominance::compute_idom() {

    // Mapping between dfn and vertex. 0 represents the entry node.
    std::unordered_map<Node*, size_t> dfn;
    std::vector<Node*> vertices;

    // Parent in the DFS tree.
    std::vector<ssize_t> parents;

    // Do a depth-first search to assign these nodes a DFN and determine their parents in DFS tree.
    {
        std::deque<std::pair<size_t, Node*>> stack { {-1, _graph.entry() }};
        while (!stack.empty()) {
            size_t parent;
            Node* node;
            std::tie(parent, node) = stack.front();
            stack.pop_front();

            auto& id = dfn[node];

            // If id == 0, then it is either the entry node, or this is a freshly encountered node.
            // As the entry node will only be visited once, id != 0 means the node is already visited.
            if (id != 0) continue;

            id = vertices.size();
            vertices.push_back(node);
            parents.push_back(parent);

            if (node->opcode() == Opcode::exit) continue;

            auto end = node->opcode() == Opcode::entry ? node : static_cast<Paired*>(node)->mate();
            for (auto value: end->values()) {

                // Skip keepalive edges.
                bool skip_exit = value.references().size() == 2;

                for (auto ref: value.references()) {
                    if (skip_exit && ref->opcode() == Opcode::exit) continue;
                    stack.push_front({id, ref});
                }
            }
        }
    }

    // Initialize variables.
    size_t count = vertices.size();
    std::vector<size_t> sdoms(count);
    std::vector<ssize_t> idoms(count, -1);
    std::vector<ssize_t> ancestors(count, -1);
    std::vector<size_t> bests(count);
    std::vector<std::vector<size_t>> buckets(count);
    for (size_t i = 0; i < count; i++) {
        sdoms[i] = i;
        bests[i] = i;
    }

    // Lengauer-Tarjan algorithm with simple eval and link.
    std::function<size_t(size_t)> eval = [&](size_t node) {
        auto ancestor = ancestors[node];
        if (ancestor == -1) return node;
        if (ancestors[ancestor] != -1) {
            eval(ancestor);
            if (sdoms[bests[node]] > sdoms[bests[ancestor]]) bests[node] = bests[ancestor];
            ancestors[node] = ancestors[ancestor];
        }
        return bests[node];
    };

    auto link = [&](size_t parent, size_t node) {
        ancestors[node] = parent;
    };

    for (size_t i = count - 1; i > 0; i--) {
        auto node = vertices[i];
        auto parent = parents[i];
        for (auto operand: node->operands()) {

            // Skip keepalive edges.
            if (node->opcode() == Opcode::exit && operand.references().size() == 2) continue;

            // Retrieve the starting node.
            auto block = operand.node();
            if (block->opcode() != Opcode::entry) block = static_cast<Paired*>(block)->mate();

            size_t pred = dfn[block];
            size_t u = eval(pred);
            if (sdoms[i] > sdoms[u]) {
                sdoms[i] = sdoms[u];
            }
        }
        buckets[sdoms[i]].push_back(i);
        link(parent, i);

        for (auto v: buckets[parent]) {
            auto u = eval(v);
            idoms[v] = sdoms[u] < sdoms[v] ? u : parent;
        }
        buckets[parent].clear();
    }

    for (size_t i = 1; i < count; i++) {
        ASSERT(idoms[i] != -1);
        if (static_cast<size_t>(idoms[i]) != sdoms[i]) {
            idoms[i] = idoms[idoms[i]];
        }

        // Turn DFN relation into relations between actual ir::Node's.
        _idom[vertices[i]] = vertices[idoms[i]];
    }
}

void Dominance::compute_ipdom() {

    // Mapping between dfn and vertex. 0 represents the exit node.
    std::unordered_map<Node*, size_t> dfn;
    std::vector<Node*> vertices;

    // Parent in the DFS tree.
    std::vector<ssize_t> parents;

    // Do a depth-first search to assign these nodes a DFN and determine their parents in DFS tree.
    {
        std::deque<std::pair<size_t, Node*>> stack { {-1, _graph.exit() }};
        while (!stack.empty()) {
            size_t parent;
            Node* node;
            std::tie(parent, node) = stack.front();
            stack.pop_front();

            auto& id = dfn[node];

            // If id == 0, then it is either the exit node, or this is a freshly encountered node.
            // As the exit node will only be visited once, id != 0 means the node is already visited.
            if (id != 0) continue;

            id = vertices.size();
            vertices.push_back(node);
            parents.push_back(parent);

            for (auto operand: node->operands()) {

                // Skip keepalive edges.
                if (id == 0 && operand.references().size() == 2) continue;

                // Retrive the starting node.
                auto block = operand.node();
                if (block->opcode() != Opcode::entry) block = static_cast<Paired*>(block)->mate();

                stack.push_front({id, block});
            }
        }
    }

    // Initialize variables.
    size_t count = vertices.size();
    std::vector<size_t> sdoms(count);
    std::vector<ssize_t> idoms(count, -1);
    std::vector<ssize_t> ancestors(count, -1);
    std::vector<size_t> bests(count);
    std::vector<std::vector<size_t>> buckets(count);
    for (size_t i = 0; i < count; i++) {
        sdoms[i] = i;
        bests[i] = i;
    }

    // Lengauer-Tarjan algorithm with simple eval and link.
    std::function<size_t(size_t)> eval = [&](size_t node) {
        auto ancestor = ancestors[node];
        if (ancestor == -1) return node;
        if (ancestors[ancestor] != -1) {
            eval(ancestor);
            if (sdoms[bests[node]] > sdoms[bests[ancestor]]) bests[node] = bests[ancestor];
            ancestors[node] = ancestors[ancestor];
        }
        return bests[node];
    };

    auto link = [&](size_t parent, size_t node) {
        ancestors[node] = parent;
    };

    for (size_t i = count - 1; i > 0; i--) {
        auto node = vertices[i];
        auto parent = parents[i];

        auto end = node->opcode() == Opcode::entry ? node : static_cast<Paired*>(node)->mate();
        for (auto value: end->values()) {

            // Skip keepalive edges.
            bool skip_exit = value.references().size() == 2;

            for (auto ref: value.references()) {
                if (skip_exit && ref->opcode() == Opcode::exit) continue;

                size_t pred = dfn[ref];

                // Unencountered node in DFS. This indicates an infinite loop, in this case we abort post-dominator
                // tree construction.
                if (pred == 0 && ref->opcode() != Opcode::exit) return;

                size_t u = eval(pred);
                if (sdoms[i] > sdoms[u]) {
                    sdoms[i] = sdoms[u];
                }
            }
        }
        buckets[sdoms[i]].push_back(i);
        link(parent, i);

        for (auto v: buckets[parent]) {
            auto u = eval(v);
            idoms[v] = sdoms[u] < sdoms[v] ? u : parent;
        }
        buckets[parent].clear();
    }

    for (size_t i = 1; i < count; i++) {
        ASSERT(idoms[i] != -1);
        if (static_cast<size_t>(idoms[i]) != sdoms[i]) {
            idoms[i] = idoms[idoms[i]];
        }

        // Turn DFN relation into relations between actual ir::Node's.
        _ipdom[vertices[i]] = vertices[idoms[i]];
    }
}

void Dominance::compute_df() {
    for (auto node: _blocks) {

        // Nodes in dominance frontier must have multiple predecessor.
        if (node->operand_count() == 1) continue;

        auto idom = _idom[node];
        for (auto operand: node->operands()) {
            auto runner = operand.node();
            if (runner->opcode() != Opcode::entry) runner = static_cast<Paired*>(runner)->mate();

            // Walk up the DOM tree until the idom is met.
            while (runner != idom) {
                ASSERT(runner);
                _df[runner].insert(node);
                runner = _idom[runner];
            }
        }
    }

void Dominance::compute_pdf() {

    // Abort if post-dominator tree does not exist.
    if (_ipdom.empty()) return;

    for (auto node: _blocks) {

        // Nodes in post-dominance frontier must have multiple successor.
        auto end = static_cast<Paired*>(node)->mate();
        if (end->value_count() == 1) continue;

        auto ipdom = _ipdom[node];
        for (auto value: end->values()) {

            // Skip keepalive edges.
            bool skip_exit = value.references().size() == 2;

            for (auto runner: value.references()) {
                if (skip_exit && runner->opcode() == Opcode::exit) continue;

                while (runner != ipdom) {
                    ASSERT(runner);
                    _pdf[runner].insert(node);
                    runner = _ipdom[runner];
                }
            }
        }
    }
}
}

}
