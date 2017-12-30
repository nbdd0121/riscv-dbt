#include "emu/mmu.h"
#include "emu/state.h"
#include "ir/builder.h"
#include "ir/pass.h"
#include "util/functional.h"

namespace ir::pass {

void Lowering::after(Node* node) {

    // We perform target-independent lowering here. After lowering, load/store_memory represents loading and storing
    // represents host address space instead of guest's. For paging MMU, memory operations are translated to helper
    // function calls.

    Builder builder { *_graph };
    switch (node->opcode()) {
        case Opcode::load_memory: {

            // In this case lowering is not needed.
            if (dynamic_cast<emu::Id_mmu*>(_state.mmu.get())) {
                break;
            }

            if (emu::Flat_mmu* flat_mmu = dynamic_cast<emu::Flat_mmu*>(_state.mmu.get())) {
                auto memory_base = builder.constant(Type::i64, reinterpret_cast<uintptr_t>(flat_mmu->memory_));
                auto computed_address = builder.arithmetic(Opcode::add, node->operand(1), memory_base);
                node->operand_set(1, computed_address);

            } else {
                auto output = node->value(1);

                uintptr_t func;
                switch (output.type()) {
                    case Type::i8: func = reinterpret_cast<uintptr_t>(
                        AS_FUNCTION_POINTER(&emu::Paging_mmu::load_memory<uint8_t>)
                    ); break;
                    case Type::i16: func = reinterpret_cast<uintptr_t>(
                        AS_FUNCTION_POINTER(&emu::Paging_mmu::load_memory<uint16_t>)
                    ); break;
                    case Type::i32: func = reinterpret_cast<uintptr_t>(
                        AS_FUNCTION_POINTER(&emu::Paging_mmu::load_memory<uint32_t>)
                    ); break;
                    case Type::i64: func = reinterpret_cast<uintptr_t>(
                        AS_FUNCTION_POINTER(&emu::Paging_mmu::load_memory<uint64_t>)
                    ); break;
                    default: ASSERT(0);
                }

                auto mmu_arg = builder.constant(Type::i64, reinterpret_cast<uintptr_t>(_state.mmu.get()));
                auto call_node = _graph->manage(new Call(
                    func, false, {Type::memory, output.type()}, {node->operand(0), mmu_arg, node->operand(1)}
                ));

                replace(node->value(0), call_node->value(0));
                replace(output, call_node->value(1));
            }
            break;
        }
        case Opcode::store_memory: {

            // In this case lowering is not needed.
            if (dynamic_cast<emu::Id_mmu*>(_state.mmu.get())) {
                break;
            }

            if (emu::Flat_mmu* flat_mmu = dynamic_cast<emu::Flat_mmu*>(_state.mmu.get())) {
                auto memory_base = builder.constant(Type::i64, reinterpret_cast<uintptr_t>(flat_mmu->memory_));
                auto computed_address = builder.arithmetic(Opcode::add, node->operand(1), memory_base);
                node->operand_set(1, computed_address);

            } else {
                auto value = node->operand(2);

                uintptr_t func;
                switch (value.type()) {
                    case Type::i8: func = reinterpret_cast<uintptr_t>(
                        AS_FUNCTION_POINTER(&emu::Paging_mmu::store_memory<uint8_t>)
                    ); break;
                    case Type::i16: func = reinterpret_cast<uintptr_t>(
                        AS_FUNCTION_POINTER(&emu::Paging_mmu::store_memory<uint16_t>)
                    ); break;
                    case Type::i32: func = reinterpret_cast<uintptr_t>(
                        AS_FUNCTION_POINTER(&emu::Paging_mmu::store_memory<uint32_t>)
                    ); break;
                    case Type::i64: func = reinterpret_cast<uintptr_t>(
                        AS_FUNCTION_POINTER(&emu::Paging_mmu::store_memory<uint64_t>)
                    ); break;
                    default: ASSERT(0);
                }

                auto mmu_arg = builder.constant(Type::i64, reinterpret_cast<uintptr_t>(_state.mmu.get()));
                auto call_node = _graph->manage(new Call(
                    func, false, {Type::memory}, {node->operand(0), mmu_arg, node->operand(1), value}
                ));

                replace(node->value(0), call_node->value(0));
            }
            break;
        }
        default:
            break;
    }
}

}
