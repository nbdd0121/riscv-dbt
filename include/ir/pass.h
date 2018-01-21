#ifndef IR_PASS_H
#define IR_PASS_H

#include <ostream>
#include <unordered_set>

#include "ir/node.h"

namespace emu {
    struct State;
};

namespace ir::pass {

class Pass {
public:
    static void replace(Value oldvalue, Value newvalue);

protected:
    Graph* _graph;

private:
    void run_recurse(Node* node);

protected:

    // Before visiting the tree.
    virtual void start() {}
    // After visiting the tree.
    virtual void finish() {}
    // Before visiting children of the node. Returning true will abort children visit.
    virtual bool before(Node*) { return false; }
    // After all children has been visited.
    virtual void after(Node*) {}

public:
    void run_on(Graph& graph, Node* node);
    void run(Graph& graph) { run_on(graph, graph.exit()); }
};

class Dot_printer: public Pass {
public:
    static const char* opcode_name(uint16_t opcode);
    static const char* type_name(Type type);

protected:
    virtual void write_node_content(std::ostream& stream, Node* node);
    virtual void start() override;
    virtual void finish() override;
    virtual void after(Node* node) override;
};

class Register_access_elimination: public Pass {
private:
    std::vector<Node*> last_load;
    std::vector<Node*> last_store;
    // Do not use std::vector<bool> as we don't need its space optimization.
    std::vector<char> has_store_after_exception;

    Value last_exception;
    Value last_effect;

    // Whether the pass is allowed to move store_register after memory operations.
    bool _strict;

public:
    Register_access_elimination(int regcount, bool strict = true):
        last_load(regcount), last_store(regcount), has_store_after_exception(regcount), _strict{strict} {}

private:
    Value merge_memory(std::vector<Value> values);

protected:
    virtual void after(Node* node) override;
};

class Local_value_numbering: public Pass {
private:
    struct Hash {
        size_t operator ()(Value value) const noexcept;
    };

    struct Equal_to {
        bool operator ()(Value a, Value b) const noexcept;
    };

private:
    std::unordered_set<Value, Hash, Equal_to> _set;

    static uint64_t sign_extend(Type type, uint64_t value);
    static uint64_t zero_extend(Type type, uint64_t value);
    static uint64_t cast(Type type, Type oldtype, bool sext, uint64_t value);
    static uint64_t binary(Type type, uint16_t opcode, uint64_t l, uint64_t r);

    Value new_constant(Type type, uint64_t const_value);
    void replace_with_constant(Value value, uint64_t const_value);
    void lvn(Value value);

protected:
    virtual void after(Node* node) override;
};

// Target-independent lowering pass.
class Lowering: public Pass {
private:
    emu::State& _state;

public:
    Lowering(emu::State& state): _state{state} {}

protected:
    virtual void after(Node* node) override;
};

}

#endif
