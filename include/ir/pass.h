#ifndef IR_PASS_H
#define IR_PASS_H

#include "ir/instruction.h"

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
    void run(Graph& graph);
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

}

#endif
