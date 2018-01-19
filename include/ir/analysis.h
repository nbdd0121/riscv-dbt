#include <unordered_map>
#include <unordered_set>

#include "ir/node.h"

namespace ir {

class Dominance {
    Graph& _graph;
    std::vector<Node*> _blocks;

    // Immediate dominators of nodes.
    std::unordered_map<Node*, Node*> _idom;

    // Immediate post-dominators of nodes.
    std::unordered_map<Node*, Node*> _ipdom;

    // Dominance frontier of nodes.
    std::unordered_map<Node*, std::unordered_set<Node*>> _df;

    // Post-dominance frontier of nodes.
    std::unordered_map<Node*, std::unordered_set<Node*>> _pdf;

public:
    Dominance(Graph& graph): _graph{graph} {
        compute_idom();
        compute_ipdom();
        compute_blocks();
        compute_df();
        compute_pdf();
    }

private:
    void compute_blocks();
    void compute_idom();
    void compute_ipdom();
    void compute_df();
    void compute_pdf();

};

}
