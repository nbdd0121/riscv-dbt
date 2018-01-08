#include <algorithm>

#include "ir/node.h"
#include "ir/pass.h"

namespace ir::pass {

Value Register_access_elimination::get_tail_jmp_pc(Value control, uint16_t pc_regnum) {
    ASSERT(control.references().size() == 1);
    auto target = *control.references().begin();

    // Not tail position
    if (target->opcode() != ir::Opcode::end) return {};

    auto last_mem = control.node()->operand(0);
    if (last_mem.opcode() == ir::Opcode::fence) {
        for (auto operand: last_mem.node()->operands()) {
            if (operand.opcode() == ir::Opcode::store_register &&
                static_cast<ir::Register_access*>(operand.node())->regnum() == pc_regnum) {

                return operand.node()->operand(1);
            }
        }

    } else if (last_mem.opcode() == ir::Opcode::store_register &&
               static_cast<ir::Register_access*>(last_mem.node())->regnum() == pc_regnum) {

        return last_mem.node()->operand(1);

    }

    return {};
}

Value Register_access_elimination::merge_memory(std::vector<Value> values) {
    ASSERT(!values.empty());
    if (values.size() == 1) return values[0];
    return _graph->manage(new Node(Opcode::fence, {Type::memory}, std::move(values)))->value(0);
}

void Register_access_elimination::after(Node* node) {

    // load_register needs to happen after previous stores.
    // store_register needs to happen after previous loads, stores, nodes w/ exceptions.
    // nodes w/ exceptions need to happen after previous stores, nodes w/ exceptions.
    // sequencing with nodes w/ side-effects must be kept.

    // To avoid storing about all previous nodes, we make the following trivial observations:
    // All stores will depend on previous stores, so only last store is necessary. Therefore we only keep `last_store`.
    // Similarly, all nodes w/ exceptions will need to depend on previous nodes w/ exceptions, so we only
    // keep `last_exception`.

    // To avoid storing `all previous loads`, we observe that register loads without storing in between are redundant.
    // Therefore we will replace all register loads with previous register loads, provided that no storing happens in
    // between. After this operation, we will only need to store `last_load`. To maintain soundness we clear
    // `last_load` for each store node.

    // After the above optimizations, there are still redundancies. A store node after another store node need not
    // depend on previous nodes w/ exceptions. We keep `has_store_after_exception` for this purpose.

    // In non-strict mode many limitations are lifted. store_register and last_exception is no longer required to
    // depend on each other.

    switch (node->opcode()) {
        case Opcode::block: {
            ASSERT(!last_effect);
            last_effect = node->value(0);
            break;
        }
        case Opcode::load_register: {
            uint16_t regnum = static_cast<Register_access*>(node)->regnum();

            // As mentioned above, replace register load with previous load. With all optimizations we applied, this
            // replacement is necessary to make other transformations sound.
            if (last_load[regnum]) {
                replace(node->value(1), last_load[regnum]->value(1));
                break;
            }

            // Eliminate load immediately after store
            if (last_store[regnum]) {
                replace(node->value(1), last_store[regnum]->operand(1));
                break;
            }

            node->operand_set(0, last_effect);

            last_load[regnum] = node;
            break;
        }
        case Opcode::store_register: {
            uint16_t regnum = static_cast<Register_access*>(node)->regnum();

            std::vector<Value> dependencies;

            // If the load is after previous store, and the store is after previous exception, the depending solely on
            // last_load is sufficient. Otherwise we will in addition need to depend on last_store or last_exception.

            if (last_load[regnum]) dependencies.push_back(last_load[regnum]->value(0));

            if (has_store_after_exception[regnum]) {
                if (dependencies.empty()) {

                    // In this case we have store after previous node w/ exceptions, and there is no load after
                    // the store. We will only depend on last store in this case. Note that this store is not a
                    // dependency of other nodes, so we can also eliminate the store by depending directly on
                    // its dependencies.
                    dependencies.push_back(last_store[regnum]->operand(0));
                }

            } else {
                if (_strict && last_exception) dependencies.push_back(last_exception);
            }

            if (dependencies.empty()) {
                ASSERT(last_effect);
                dependencies.push_back(last_effect);
            }

            node->operand_set(0, merge_memory(dependencies));

            last_load[regnum] = nullptr;
            last_store[regnum] = node;
            has_store_after_exception[regnum] = 1;
            break;
        }
        case Opcode::load_memory:
        case Opcode::store_memory:  {
            std::vector<Value> dependencies;

            // Nodes w/ exceptions depend on all previous stores.
            if (_strict) {
                for (size_t regnum = 0; regnum < last_load.size(); regnum++) {
                    if (has_store_after_exception[regnum]) dependencies.push_back(last_store[regnum]->value(0));
                    has_store_after_exception[regnum] = 0;
                }
            }

            // We need to depend on last_exception or last_effect if we do not depend on them indirectly.
            if (dependencies.empty()) {
                if (last_exception) {
                    dependencies.push_back(last_exception);
                } else {
                    ASSERT(last_effect);
                    dependencies.push_back(last_effect);
                }
            }

            node->operand_set(0, merge_memory(dependencies));

            last_exception = node->value(0);
            break;
        }
        case Opcode::call:
        case Opcode::i_if:
        case Opcode::jmp: {
            std::vector<Value> dependencies;

            bool need_last_exception = true;
            for (size_t regnum = 0; regnum < last_load.size(); regnum++) {

                // The following logic is similar to store_register.
                if (last_load[regnum]) dependencies.push_back(last_load[regnum]->value(0));
                if (has_store_after_exception[regnum]) {
                    ASSERT(!last_load[regnum]);
                    dependencies.push_back(last_store[regnum]->value(0));
                    need_last_exception = false;
                }

                has_store_after_exception[regnum] = 0;
                last_load[regnum] = nullptr;
                last_store[regnum] = nullptr;
            }

            if ((!_strict || need_last_exception) && last_exception) dependencies.push_back(last_exception);
            if (dependencies.empty()) {
                ASSERT(last_effect);
                dependencies.push_back(last_effect);
            }

            node->operand_set(0, merge_memory(dependencies));

            last_exception = {};

            if (node->opcode() == Opcode::call) {
                last_effect = node->value(0);
            } else {
                // if and jmp node will turn memory dependency into control, so last_effect needs to be cleared.
                last_effect = {};
            }
            break;
        }
        default: break;
    }
}

}
