#include <unordered_map>
#include <unordered_set>

#include "ir/node.h"

namespace ir {

namespace analysis {

// Helper function for control flow related analysis.
class Block {
public:
    // Get the real target of a control. Ignore keepalive edges.
    static Node* get_target(Value control);

    // Given a control, verify if it is a tail jump (jump to exit), and whether the pc of the next block is a known
    // value. The value of pc of next block will be returned, or null will be returned if it is not a tail jump, or
    // the value of pc is unknown.
    static Value get_tail_jmp_pc(Value control, uint16_t pc_regnum);

};

}

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
