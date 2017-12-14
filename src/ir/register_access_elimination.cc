#include <algorithm>

#include "ir/instruction.h"
#include "ir/pass.h"

namespace ir::pass {

Instruction* Register_access_elimination::dependency(std::vector<Instruction*>&& dep) {
    if (dep.empty()) return nullptr;
    if (dep.size() == 1) return dep[0];
    return _graph->manage(new Instruction(Type::none, Opcode::fence, std::move(dep)));
}

void Register_access_elimination::after(Instruction* inst) {

    // load_register needs to happen after previous stores.
    // store_register needs to happen after previous loads, stores, instructions w/ exceptions.
    // instructions w/ exceptions need to happen after previous stores, instructions w/ exceptions.
    // sequencing with instructions w/ side-effects must be kept.

    // To avoid storing about all previous instructions, we make the following trivial observations:
    // All stores will depend on previous stores, so only last store is necessary. Therefore we only keep `last_store`.
    // Similarly, all instructions w/ exceptions will need to depend on previous instructions w/ exceptions, so we only
    // keep `last_exception`.

    // To avoid storing `all previous loads`, we observe that register loads without storing in between are redundant.
    // Therefore we will replace all register loads with previous register loads, provided that no storing happens in
    // between. After this operation, we will only need to store `last_load`. To maintain soundness we clear
    // `last_load` for each store instruction.

    // After the above optimizations, there are still redundancies. A store instruction after another store instruction
    // need not depend on previous instructions w/ exceptions. We keep `has_store_after_exception` for this purpose.

    switch (inst->opcode()) {
        case Opcode::block: {
            ASSERT(last_effect == nullptr);
            last_effect = inst;
            break;
        }
        case Opcode::load_register: {
            int regnum = inst->attribute();

            // As mentioned above, replace register load with previous load. With all optimizations we applied, this
            // replacement is necessary to make other transformations sound.
            if (last_load[regnum]) {
                replace(inst, last_load[regnum]);
                break;
            }

            // Otherwise this is the first load after side-effect or last store. Calculate the dependency here.
            auto dep = last_store[regnum];

            // Eliminate load immediately after store
            if (dep && dep->opcode() == Opcode::store_register) {
                replace(inst, dep->operand(1));
                break;
            }

            if (!dep) dep = last_effect;
            inst->operand_set(0, dep);

            last_load[regnum] = inst;
            break;
        }
        case Opcode::store_register: {
            int regnum = inst->attribute();

            auto dep = last_load[regnum];

            // If the load is after previous store, and the store is after previous exception, the depending solely on
            // last_load is sufficient. Otherwise we will in addition need to depend on last_store or last_exception.

            if (!has_store_after_exception[regnum]) {
                if (!dep) {
                    dep = last_exception;
                } else if (last_exception) {
                    dep = _graph->manage(new Instruction(Type::none, Opcode::fence, {dep, last_exception}));
                }
            } else if (!dep) {
                // In this case we have store after previous instruction w/ exceptions, and there is no load after the
                // store. We will only depend on last store in this case.
                dep = last_store[regnum];
                // Eliminate store after store.
                dep = dep->operand(0);
            }

            if (!dep) dep = last_effect;
            inst->operand_set(0, dep);

            last_load[regnum] = nullptr;
            last_store[regnum] = inst;
            has_store_after_exception[regnum] = 1;
            break;
        }
        case Opcode::load_memory:
        case Opcode::store_memory:  {
            std::vector<Instruction*> dependencies;

            // Instructions w/ exceptions depend on all previous stores.
            for (size_t regnum = 0; regnum < last_load.size(); regnum++) {
                if (has_store_after_exception[regnum]) dependencies.push_back(last_store[regnum]);
                has_store_after_exception[regnum] = 0;
            }

            auto dep = dependency(std::move(dependencies));
            if (!dep) dep = last_exception;
            if (!dep) dep = last_effect;
            inst->operand_set(0, dep);

            last_exception = inst;
            break;
        }
        case Opcode::emulate:
        case Opcode::jmp: {
            std::vector<Instruction*> dependencies;

            bool need_last_exception = true;
            for (size_t regnum = 0; regnum < last_load.size(); regnum++) {
                
                // The following logic is similar to store_register.
                if (last_load[regnum]) dependencies.push_back(last_load[regnum]);
                if (has_store_after_exception[regnum]) {
                    if (!last_load[regnum]) dependencies.push_back(last_store[regnum]);
                    need_last_exception = false;
                }

                has_store_after_exception[regnum] = 0;
                last_load[regnum] = nullptr;
                last_store[regnum] = nullptr;
            }

            if (need_last_exception && last_exception) dependencies.push_back(last_exception); 
            auto dep = dependency(std::move(dependencies));
            if (!dep) dep = last_effect;
            inst->operand_set(0, dep);

            last_exception = nullptr;
            last_effect = inst;
            break;
        }
        default: break;
    }
}

}
