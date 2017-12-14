#include "ir/instruction.h"
#include "ir/pass.h"

#include "util/assert.h"
#include "util/format.h"

namespace ir::pass {

const char* Printer::opcode_name(Opcode opcode) {
    switch (opcode) {
#define CASE(name) case Opcode::name: return #name;
        CASE(start)
        CASE(end)
        CASE(block)
        case Opcode::i_if: return "if";
        CASE(if_true)
        CASE(if_false)
        CASE(jmp)
        CASE(constant)
        CASE(cast)
        CASE(fence)
        CASE(load_register)
        CASE(store_register)
        CASE(load_memory)
        CASE(store_memory)
        CASE(emulate)
        CASE(add)
        CASE(sub)
        case Opcode::i_xor: return "xor";
        case Opcode::i_or: return "or";
        case Opcode::i_and: return "and";
        CASE(shl)
        CASE(shr)
        CASE(sar)
        CASE(eq)
        CASE(ne)
        CASE(lt)
        CASE(ge)
        CASE(ltu)
        CASE(geu)
        CASE(neg)
        case Opcode::i_not: return "not";
#undef CASE
        default: return "(unknown)";
    }
}

const char* Printer::type_name(Type type) {
    switch (type) {
#define CASE(name) case Type::name: return #name;
        CASE(none)
        CASE(i1)
        CASE(i8)
        CASE(i16)
        CASE(i32)
        CASE(i64)
#undef CASE
        default: return "(unknown)";
    }
}

void Printer::after(Instruction* inst) {

    // Display the type and number of the instruction. If the instruction does not return value
    if (inst->type() != Type::none) {
        inst->scratchpad(_index++);
        util::log("{} %{} = ", type_name(inst->type()), inst->scratchpad());
    } else {
        // In debugging mode, set the index of an instruction returning none to -1 to ease debugging.
        if (ASSERT_STRATEGY != ASSERT_STRATEGY_ASSUME) {
            inst->scratchpad(-1);
        }
    }

    // As a linear IR printer, we omit some control node.
    // TODO: Here are some horrible hacks. Once we actually introduce multiple blocks, this needs major change.
    switch (inst->opcode()) {
        case Opcode::start:
        case Opcode::end:
        case Opcode::block:
        case Opcode::jmp:
            return;
        default: break;
    }

    // As we does not output side-effect dependency, fence is omitted.
    if (inst->opcode() == Opcode::fence) return;

    std::clog << opcode_name(inst->opcode());

    // In the linear IR printer, we does not output side-effect dependency relations between instructions. The switch
    // below will set it to true for those instructions w/ side-effect dependencies so we can skip them later.
    bool has_dependency = false;

    // Output attributes for those special instructions.
    switch (inst->opcode()) {
        case Opcode::constant:
            std::clog << ' ' << static_cast<int64_t>(inst->attribute());
            break;
        case Opcode::cast:
            if (inst->attribute()) std::clog << " sext";
            break;
        case Opcode::load_register:
            has_dependency = true;
            std::clog << " r" << inst->attribute();
            break;
        case Opcode::store_register:
            has_dependency = true;
            std::clog << " r" << inst->attribute() << ',';
            break;
        case Opcode::load_memory:
        case Opcode::store_memory:
        case Opcode::emulate:
            has_dependency = true;
            break;
        default: break;
    }

    bool first = true;
    for (auto operand: inst->operands()) {
        if (first) {
            // Skip the first operand if it is the side-effect dependency.
            if (has_dependency) {
                has_dependency = false;
                continue;
            }
            first = false;
            std::clog << ' ';
        } else {
            std::clog << ", ";
        }
        std::clog << "%" << operand->scratchpad();
    }

    std::clog << std::endl;
}

}
