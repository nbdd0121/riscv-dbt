#ifndef IR_PASS_H
#define IR_PASS_H

#include <unordered_set>

#include "ir/instruction.h"

namespace riscv {
    struct Context;
};

namespace ir::pass {

class Pass {
public:
    static void replace(Instruction* oldnode, Instruction* newnode);

protected:
    Graph* _graph;

private:
    void run_recurse(Instruction* inst);

protected:

    // Before visiting the tree.
    virtual void start() {}
    // After visiting the tree.
    virtual void finish() {}
    // Before visiting children of the instruction. Returning true will abort children visit.
    virtual bool before(Instruction*) { return false; }
    // After all children has been visited.
    virtual void after(Instruction*) {}

public:
    void run_on(Graph& graph, Instruction* inst);
    void run(Graph& graph) { run_on(graph, graph.root()); }
};

class Printer: public Pass {
public:
    static const char* opcode_name(Opcode opcode);
    static const char* type_name(Type type);

protected:
    // Used for numbering the output of instructions.
    uint64_t _index;
    virtual void start() override { _index = 0; }
    virtual void after(Instruction* inst) override;
};

class Dot_printer: public Printer {
protected:
    virtual void start() override;
    virtual void finish() override;
    virtual bool before(Instruction* inst) override;
    virtual void after(Instruction* inst) override;
};

class Register_access_elimination: public Printer {
private:
    std::vector<Instruction*> last_load;
    std::vector<Instruction*> last_store;
    // Do not use std::vector<bool> as we don't need its space optimization.
    std::vector<char> has_store_after_exception;

    Instruction* last_exception = nullptr;
    Instruction* last_effect = nullptr;
public:
    Register_access_elimination(int regcount):
        last_load(regcount), last_store(regcount), has_store_after_exception(regcount) {}
private:
    Instruction* dependency(std::vector<Instruction*>&& dep);

protected:
    virtual void after(Instruction* inst) override;
};

class Evaluator: public Pass {
public:
    static uint64_t sign_extend(Type type, uint64_t value);
    static uint64_t zero_extend(Type type, uint64_t value);
    static uint64_t cast(Type type, Type oldtype, bool sext, uint64_t value);
    static uint64_t binary(Type type, Opcode opcode, uint64_t l, uint64_t r);

private:
    // TODO: Make the evaluator more generic.
    riscv::Context* _ctx;

public:
    Evaluator(riscv::Context* ctx): _ctx {ctx} {};

    virtual void after(Instruction* inst) override;
};

class Local_value_numbering: public Pass {
private:
    struct Hash {
        size_t operator ()(Instruction* inst) const noexcept;
    };
    struct Equal_to {
        bool operator ()(Instruction* a, Instruction* b) const noexcept;
    };
public:
    static void replace_with_constant(Instruction* inst, uint64_t value);

private:
    std::unordered_set<Instruction*, Hash, Equal_to> _set;

protected:
    virtual void after(Instruction* inst) override;
};

}

#endif
