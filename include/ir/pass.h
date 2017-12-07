#ifndef IR_PASS_H
#define IR_PASS_H

#include "ir/instruction.h"

namespace ir::pass {

class Pass {
private:
    void run_recurse(Instruction* inst);

protected:
    // Before visiting children of the instruction. Returning true will abort children visit.
    virtual bool before(Instruction*) { return false; }
    // After all children has been visited.
    virtual void after(Instruction*) {}

public:
    void run(std::vector<ir::Instruction*>& buffer);
};

}

#endif
