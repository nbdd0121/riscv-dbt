#include <unordered_map>
#include <unordered_set>

#include "ir/node.h"

namespace ir {

class Dominance {
    Graph& _graph;
    std::vector<Node*> _blocks;

    // Immediate dominators of nodes.
    std::unordered_map<Node*, Node*> _idom;

    // Dominance frontier of nodes.
    std::unordered_map<Node*, std::unordered_set<Node*>> _df;

public:
    Dominance(Graph& graph): _graph{graph} {
        compute_idom();
        compute_blocks();
        compute_df();
    }

private:
    void compute_blocks();
    void compute_idom();
    void compute_df();

};

}