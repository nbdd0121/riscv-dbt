#include "emu/mmu.h"
#include "ir/instruction.h"
#include "ir/pass.h"
#include "riscv/context.h"
#include "riscv/instruction.h"
#include "util/memory.h"

namespace ir::pass {

// Sign-extend value of type to i64
uint64_t Evaluator::sign_extend(Type type, uint64_t value) {
    switch (type) {
        case Type::i1: return value ? 1 : 0;
        case Type::i8: return static_cast<int64_t>(static_cast<int8_t>(value));
        case Type::i16: return static_cast<int64_t>(static_cast<int16_t>(value));
        case Type::i32: return static_cast<int64_t>(static_cast<int32_t>(value));
        case Type::i64: return value;
        default: ASSERT(0);
    }
}

// Zero-extend value of type to i64
uint64_t Evaluator::zero_extend(Type type, uint64_t value) {
    switch (type) {
        case Type::i1: return value ? 1 : 0;
        case Type::i8: return static_cast<uint8_t>(value);
        case Type::i16: return static_cast<uint16_t>(value);
        case Type::i32: return static_cast<uint32_t>(value);
        case Type::i64: return value;
        default: ASSERT(0);
    }
}

// Evaluate cast node.
uint64_t Evaluator::cast(Type type, Type oldtype, bool sext, uint64_t value) {
    // For signed upcast, it can be represented as sign-extend to 64-bit and downcast.
    // For unsigned upcast, it can be represented as zero-extend to 64-bit and downcast.
    // For downcast, sign-extending or zero-extending makes no difference.
    // We choose to express all values using 64-bit number, sign-extended, as this representation allows comparision
    // without knowing the type of the value.
    if (sext) {
        return sign_extend(type, value);
    } else {
        return sign_extend(type, zero_extend(oldtype, value));
    }
}

// Evaluate binary operations.
uint64_t Evaluator::binary(Type type, Opcode opcode, uint64_t l, uint64_t r) {
    switch (opcode) {
        case Opcode::add: return sign_extend(type, l + r);
        case Opcode::sub: return sign_extend(type, l - r);
        // Bitwise operations will preserve the sign-extension.
        case Opcode::i_xor: return l ^ r;
        case Opcode::i_or: return l | r;
        case Opcode::i_and: return l & r;
        case Opcode::shl: return sign_extend(type, l << (r & (get_type_size(type) - 1)));
        // To maintain correctness, convert to zero-extension, perform operation, then convert back.
        case Opcode::shr: return sign_extend(type, zero_extend(type, l) >> (r & (get_type_size(type) - 1)));
        case Opcode::sar: return static_cast<int64_t>(l) >> (r & (get_type_size(type) - 1));
        case Opcode::eq: return l == r;
        case Opcode::ne: return l != r;
        // All comparisions will work with sign-extension (which is the reason sign-extension is chosen).
        case Opcode::lt: return static_cast<int64_t>(l) < static_cast<int64_t>(r);
        case Opcode::ge: return static_cast<int64_t>(l) >= static_cast<int64_t>(r);
        case Opcode::ltu: return l < r;
        case Opcode::geu: return l >= r;
        default: ASSERT(0);
    }
}

void Evaluator::after(Instruction* inst) {
    uint64_t result = 0;
    auto opcode = inst->opcode();
    switch (opcode) {
        case Opcode::i_if:
        case Opcode::jmp:
            break;
        case Opcode::constant:
            result = inst->attribute();
            break;
        case Opcode::cast:
            result = cast(inst->type(), inst->operand(0)->type(), inst->attribute(), inst->operand(0)->scratchpad());
            break;
        case Opcode::load_register:
            // Need to use util::read_as to be standard-compliant (otherwise we may access array out of bound).
            result = util::read_as<uint64_t>(reinterpret_cast<uint64_t*>(_ctx->registers) + inst->attribute());
            break;
        case Opcode::store_register:
            util::write_as<uint64_t>(
                reinterpret_cast<uint64_t*>(_ctx->registers) + inst->attribute(),
                inst->operand(0)->scratchpad()
            );
            break;
        case Opcode::load_memory: {
            uint64_t address = inst->operand(0)->scratchpad();
            switch (inst->type()) {
                case Type::i8: result = sign_extend(Type::i8, _ctx->mmu->load_memory<uint8_t>(address)); break;
                case Type::i16: result = sign_extend(Type::i16, _ctx->mmu->load_memory<uint16_t>(address)); break;
                case Type::i32: result = sign_extend(Type::i32, _ctx->mmu->load_memory<uint32_t>(address)); break;
                case Type::i64: result = _ctx->mmu->load_memory<uint64_t>(address); break;
                default: ASSERT(0);
            }
            break;
        }
        case Opcode::store_memory: {
            uint64_t address = inst->operand(0)->scratchpad();
            uint64_t value = inst->operand(1)->scratchpad();
            switch (inst->operand(1)->type()) {
                case Type::i8: _ctx->mmu->store_memory<uint8_t>(address, value); break;
                case Type::i16: _ctx->mmu->store_memory<uint16_t>(address, value); break;
                case Type::i32: _ctx->mmu->store_memory<uint32_t>(address, value); break;
                case Type::i64: _ctx->mmu->store_memory<uint64_t>(address, value); break;
                default: ASSERT(0);
            }
            break;
        }
        case Opcode::emulate: {
            riscv::Instruction rinst;
            util::write_as<uint64_t>(&rinst, inst->attribute());
            riscv::step(_ctx, rinst);
            break;
        }
        case Opcode::neg:
            result = sign_extend(inst->type(), -inst->operand(0)->scratchpad());
            break;
        case Opcode::i_not:
            result = ~inst->operand(0)->scratchpad();
            break;
        default: {
            ASSERT(is_binary_opcode(opcode));
            uint64_t l = inst->operand(0)->scratchpad();
            uint64_t r = inst->operand(1)->scratchpad();
            result = binary(inst->type(), opcode, l, r);
            break;
        }
    }
    inst->scratchpad(result);
}

void Evaluator::run(Graph& graph) {
    auto start = graph.start();
    ASSERT(start->dependants().size() == 1);
    auto block = *start->dependants().begin();

    // While the control does not reach the end node.
    while (block != graph.root()) {
        ASSERT(block->opcode() == ir::Opcode::block);

        // Use attribute.pointer to find the last node of the block.
        auto end = static_cast<ir::Instruction*>(block->attribute_pointer());

        // Evaluate the block.
        run_on(graph, end);

        if (end->opcode() == ir::Opcode::i_if) {

            // Get the result of comparision.
            bool result = end->operand(0)->scratchpad();

            for (auto ref: end->dependants()) {
                bool expected;
                if (ref->opcode() == ir::Opcode::if_true) expected = true;
                else if (ref->opcode() == ir::Opcode::if_false) expected = false;
                else ASSERT(0);

                if (result == expected) {
                    end = ref;
                    break;
                }
            }
        } else {
            ASSERT(end->opcode() == ir::Opcode::jmp);
        }

        block = *end->dependants().begin();
    }
}

}
