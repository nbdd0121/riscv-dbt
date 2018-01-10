#include <deque>
#include <functional>

#include "ir/analysis.h"

namespace ir {

void Dominance::compute_blocks() {
    std::deque<Node*> queue { *_graph.start()->value(0).references().begin() };
    while (!queue.empty()) {
        auto node = queue.front();
        queue.pop_front();

        // Already visited.
        if (std::find(_blocks.begin(), _blocks.end(), node) != _blocks.end()) continue;

        _blocks.push_back(node);
        auto end = static_cast<Paired*>(node)->mate();
        for (auto value: end->values()) {
            for (auto ref: value.references()) {
                if (ref->opcode() == Opcode::end) continue;
                queue.push_back(ref);
            }
        }
    }
}

void Dominance::compute_idom() {

    // Mapping between dfn and vertex. 0 represents the start node.
    std::unordered_map<Node*, size_t> dfn { {_graph.start(), 0} };
    std::vector<Node*> vertices { _graph.start() };

    // Parent in the DFS tree.
    std::vector<ssize_t> parents { -1 };

    // Do a depth-first search to assign these nodes a DFN and determine their parents in DFS tree.
    {
        std::deque<std::pair<size_t, Node*>> stack { {0, *_graph.start()->value(0).references().begin() }};
        while (!stack.empty()) {
            size_t parent;
            Node* node;
            std::tie(parent, node) = stack.front();
            stack.pop_front();

            auto& id = dfn[node];

            // If id == 0, then it is either the start node, or this is a freshly encountered node.
            // As the node cannot be the start node, it must be a fresh node.
            if (id != 0) continue;

            id = vertices.size();
            vertices.push_back(node);
            parents.push_back(parent);

            auto end = static_cast<Paired*>(node)->mate();
            for (auto value: end->values()) {
                for (auto ref: value.references()) {
                    if (ref->opcode() == Opcode::end) continue;
                    stack.push_front({id, ref});
                }
            }

            dfn[end] = id;
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
        if (ancestor == -1 || ancestors[ancestor] == -1) return node;
        size_t ancestor_best = eval(ancestor);
        if (sdoms[bests[node]] > sdoms[ancestor_best]) bests[node] = ancestor_best;
        ancestors[node] = ancestors[ancestor];
        return bests[node];
    };

    auto link = [&](size_t parent, size_t node) {
        ancestors[node] = parent;
    };

    for (size_t i = count - 1; i > 0; i--) {
        auto node = vertices[i];
        auto parent = parents[i];
        for (auto operand: node->operands()) {
            size_t pred = dfn[operand.node()];
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

void Dominance::compute_df() {
    for (auto node: _blocks) {

        // Nodes in dominance frontier must have multiple predecessor.
        if (node->operand_count() == 1) continue;

        auto idom = _idom[node];
        for (auto operand: node->operands()) {
            auto runner = operand.node();
            if (runner->opcode() != Opcode::start) runner = static_cast<Paired*>(runner)->mate();

            // Walk up the DOM tree until the idom is met.
            while (runner != idom) {
                _df[runner].insert(node);
                runner = _idom[runner];
            }
        }
    }
}

}
