#include "ir/node.h"
#include "ir/pass.h"

namespace ir::pass {

void Pass::replace(Node* oldnode, Node* newnode) {
    while (!oldnode->dependants().empty()) {
        (*oldnode->dependants().rbegin())->dependency_update(oldnode, newnode);
    }

    while (!oldnode->references().empty()) {
        (*oldnode->references().rbegin())->operand_update(oldnode, newnode);
    }
}

void Pass::run_recurse(Node* node) {
    if (node->_visited) return;
    if (before(node)) {
        node->_visited = 1;
        return;
    }

    node->_visited = 2;

    // Visit all dependencies
    for (auto dependency: node->dependencies()) run_recurse(dependency);
    for (auto operand: node->operands()) run_recurse(operand);
    after(node);
    node->_visited = 1;
}

void Pass::run_on(Graph& graph, Node* node) {
    _graph = &graph;
    for (auto node: graph._heap) {
        node->_visited = false;
    }
    start();
    run_recurse(node);
    finish();
}

}
