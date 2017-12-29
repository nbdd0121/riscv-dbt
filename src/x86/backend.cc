#include "emu/mmu.h"
#include "emu/state.h"
#include "riscv/context.h"
#include "util/format.h"
#include "util/functional.h"
#include "util/memory.h"
#include "x86/backend.h"
#include "x86/builder.h"
#include "x86/disassembler.h"
#include "x86/instruction.h"

using namespace x86::builder;

static constexpr int volatile_register[] = {0, 1, 2, 6, 7, 8, 9, 10, 11};

static uint64_t spill_area[128];

static bool is_int32(uint64_t value) {
    return static_cast<uint64_t>(static_cast<int32_t>(value)) == value;
}

static int register_id(x86::Register reg) {
    ASSERT(reg != x86::Register::none);
    return static_cast<uint8_t>(reg) & 0xF;
}

static x86::Register register_of_id(ir::Type type, int reg) {
    using namespace x86;
    switch (type) {
        case ir::Type::i1:
        case ir::Type::i8: return static_cast<Register>(reg | (reg >= 4 && reg <= 7 ? reg_gpb2 : reg_gpb));
        case ir::Type::i16: return static_cast<Register>(reg | reg_gpw);
        case ir::Type::i32: return static_cast<Register>(reg | reg_gpd);
        case ir::Type::i64: return static_cast<Register>(reg | reg_gpq);
        default: ASSERT(0);
    }
}

static bool same_location(const x86::Operand& a, const x86::Operand& b) {
    ASSERT(!a.is_immediate() && !b.is_immediate());
    if (a.is_register()) {
        if (!b.is_register()) return false;
        return register_id(a.as_register()) == register_id(b.as_register());
    }
    ASSERT(a.is_memory());
    if (!b.is_memory()) return false;
    const x86::Memory& am = a.as_memory();
    const x86::Memory& bm = b.as_memory();
    return am.base == bm.base && am.index == bm.index && am.scale == bm.scale && am.displacement == bm.displacement;
}

static x86::Operand modify_size(ir::Type type, const x86::Operand& loc) {
    using namespace x86;
    if (loc.is_register()) {
        return register_of_id(type, register_id(loc.as_register()));
    } else if (loc.is_immediate()) {
        return loc;
    } else {
        Memory mem = loc.as_memory();
        mem.size = ir::get_type_size(type) / 8;
        return mem;
    }
}

namespace x86 {

void Backend::emit(const Instruction& inst) {
    bool disassemble = _state.disassemble;
    std::byte *pc;
    if (disassemble) {
        pc = _encoder.buffer().data() + _encoder.buffer().size();
    }
    try {
        _encoder.encode(inst);
    } catch (...) {
        if (disassemble) {
            x86::disassembler::print_instruction(reinterpret_cast<uintptr_t>(pc), nullptr, 0, inst);
        }
        throw;
    }
    if (disassemble) {
        std::byte *new_pc = _encoder.buffer().data() + _encoder.buffer().size();
        x86::disassembler::print_instruction(
            reinterpret_cast<uintptr_t>(pc), reinterpret_cast<const char*>(pc), new_pc - pc, inst);
    }
}

void Backend::emit_move(ir::Type type, const Operand& dst, const Operand& src) {
    if (dst.is_memory() || src.is_memory() || type == ir::Type::i64 || type == ir::Type::i32) {
        emit(mov(dst, src));
    } else {
        // 32-bit move is shorter.
        emit(mov(modify_size(ir::Type::i32, dst), modify_size(ir::Type::i32, src)));
    }
}

Register Backend::alloc_register_no_spill(ir::Type type, Register hint) {

    // If hint is given, try to use it first
    if (hint != Register::none) {
        int hint_id = register_id(hint);
        ASSERT(!pinned[hint_id]);
        if (!register_content[hint_id]) {
            return register_of_id(type, hint_id);
        }
    }

    // Scan through all usable registers for match.
    for (int reg: volatile_register) {
        if (!pinned[reg] && !register_content[reg]) {
            return register_of_id(type, reg);
        }
    }

    // Return Register::none means no such register exist.
    return Register::none;
}

Register Backend::alloc_register(ir::Type type, Register hint) {

    // Try to allocate register with hint.
    Register reg = alloc_register_no_spill(type, hint);
    if (reg != Register::none) return reg;

    // Spill out the hint.
    if (hint != Register::none) {
        int hint_id = register_id(hint);
        ASSERT(!pinned[hint_id]);
        reg = register_of_id(type, hint_id);
        spill_register(reg);
        return reg;
    }

    for (int loc : volatile_register) {
        if (!pinned[loc]) {
            reg = register_of_id(type, loc);
            spill_register(reg);
            return reg;
        }
    }

    // Running out of registers. This should never happen.
    ASSERT(0);
}

Register Backend::alloc_register(ir::Type type, const Operand& op) {
    if (op.is_register()) return alloc_register(type, op.as_register());
    return alloc_register(type);
}

void Backend::spill_register(Register reg) {
    auto value = register_content[register_id(reg)];
    ASSERT(value);

    auto ptr = memory_location.find(value);
    if (ptr == memory_location.end()) {
        Memory mem;
        stack_size -= 8;
        mem = qword(Register::none + ((uintptr_t)spill_area - stack_size));
        mem.size = get_type_size(value.type()) / 8;
        memory_location[value] = mem;
        move_location(value, mem);
    } else {
        register_content[register_id(reg)] = {};
        location[value] = ptr->second;
    }
}

void Backend::spill_all_registers() {
    for (int reg: volatile_register) {
        if (register_content[reg]) {
            spill_register(static_cast<Register>(reg | reg_gpq));
        }
    }
}

void Backend::pin_register(Register reg) {
    pinned[register_id(reg)] = true;
}

void Backend::unpin_register(Register reg) {
    pinned[register_id(reg)] = false;
}

void Backend::move_location(ir::Value value, const Operand& loc) {
    Operand old_loc = get_location(value);

    if (old_loc.is_register()) {
        int reg = register_id(old_loc.as_register());
        ASSERT(register_content[reg] == value);
        register_content[reg] = {};
    }

    if (loc.is_register()) {
        int reg = register_id(loc.as_register());
        ASSERT(!register_content[reg]);
        register_content[reg] = value;
    }

    location[value] = loc;
    emit_move(value.type(), loc, old_loc);
}

void Backend::bind_register(ir::Value value, Register loc) {
    location[value] = loc;
    register_content[register_id(loc)] = value;
    reference_count[value] = value.references().size();
}

void Backend::ensure_register(ir::Value value, Register reg) {
    Operand loc = get_location(value);

    // If it is already in that register, then good.
    if (same_location(loc, reg)) return;

    // If the target register is already occupied, spill it.
    if (register_content[register_id(reg)]) {
        spill_register(reg);
    }

    // Move to target location.
    move_location(value, reg);
}

void Backend::decrease_reference(ir::Value value) {
    if (value.is_const()) {
        return;
    }

    int& ref = reference_count.at(value);
    ref--;

    // When reference count reaches zero the value could be wiped out.
    if (ref == 0) {
        const auto& loc = location.at(value);

        if (loc.is_register()) {
            register_content[register_id(loc.as_register())] = {};
        }

        location.erase(value);
        return;
    }

    ASSERT(ref > 0);
}

Operand Backend::get_location(ir::Value value) {
    if (value.is_const()) {
        ASSERT(is_int32(value.const_value()));
        return value.const_value();
    }
    auto ptr = location.find(value);
    ASSERT(ptr != location.end());
    return ptr->second;
}

Operand Backend::get_location_ex(ir::Value value, bool allow_mem, bool allow_imm) {
    if (value.is_const()) {
        if (allow_imm && is_int32(value.const_value())) {
            return value.const_value();
        }
        Register reg = alloc_register(value.type());
        emit(mov(reg, value.const_value()));
        return reg;
    }
    Operand loc = get_location(value);
    if (allow_mem || loc.is_register()) return loc;

    Register reg = alloc_register(value.type());
    move_location(value, reg);
    return reg;
}

Register Backend::get_register_location(ir::Value value) {
    return get_location_ex(value, false, false).as_register();
}

void Backend::emit_alu(ir::Node* node, Opcode opcode) {
    auto output = node->value(0);
    auto op0 = node->operand(0);
    auto op1 = node->operand(1);

    if (op0.is_const()) {

        // We require constant folding pass, so the other operand cannot be immediate.
        ASSERT(!op1.is_const());

        // Allocate and bind register.
        Register reg = alloc_register(output.type());
        bind_register(output, reg);

        // Move immediate to register.
        emit(mov(reg, op0.const_value()));

        // Perform operation.
        emit(binary(opcode, reg, get_location(op1)));

        // Recycle the register of op1 if possible.
        decrease_reference(op1);
        return;
    }

    // Decrease reference early, so if it is the last use we can eliminate one move.
    Operand loc0 = get_location(op0);
    decrease_reference(op0);

    // Allocate and bind register. Try to use loc0 if possible to eliminate move.
    Register reg = alloc_register(output.type(), loc0);
    bind_register(output, reg);

    // Move left operand to register.
    if (!same_location(loc0, reg)) {
        emit_move(output.type(), reg, loc0);
    }

    pin_register(reg);
    emit(binary(opcode, reg, get_location_ex(op1, true, true)));
    unpin_register(reg);
    decrease_reference(op1);
}

void Backend::emit_shift(ir::Node* node, Opcode opcode) {
    auto output = node->value(0);
    auto op0 = node->operand(0);
    auto op1 = node->operand(1);

    if (!op1.is_const()) {
        // The operand must be in CL.
        ensure_register(op1, Register::cl);
        pin_register(Register::cl);
    }

    if (op0.is_const()) {

        // We require constant folding pass, so the other operand cannot be immediate.
        ASSERT(!op1.is_const());

        // Allocate and bind register.
        Register reg = alloc_register(output.type());
        bind_register(output, reg);

        // Move immediate to register.
        emit(mov(reg, op0.const_value()));

        // Perform operation.
        emit(binary(opcode, reg, Register::cl));

    } else {

        // Decrease reference early, so if it is the last use we can eliminate one move.
        Operand loc0 = get_location(op0);
        decrease_reference(op0);

        // Allocate and bind register. Try to use loc0 if possible to eliminate move.
        Register reg = alloc_register(output.type(), loc0);
        bind_register(output, reg);

        // Move left operand to register.
        if (!same_location(loc0, reg)) {
            emit_move(output.type(), reg, loc0);
        }

        if (op1.is_const()) {
            emit(binary(opcode, reg, op1.const_value()));
        } else {
            emit(binary(opcode, reg, Register::cl));
        }
    }

    if (!op1.is_const()) {
        unpin_register(Register::cl);
        decrease_reference(op1);
    }
}

void Backend::emit_unary(ir::Node* node, Opcode opcode) {
    auto output = node->value(0);
    auto op = node->operand(0);

    ASSERT(!op.is_const());

    // Decrease reference early, so if it is the last use we can eliminate one move.
    Operand loc = get_location(op);
    decrease_reference(op);

    // Allocate and bind register. Try to use loc if possible to eliminate move.
    Register reg = alloc_register(output.type(), loc);
    bind_register(output, reg);

    // Move left operand to register.
    if (!same_location(loc, reg)) {
        emit_move(output.type(), reg, loc);
    }

    emit(unary(opcode, reg));
}

Condition_code Backend::emit_compare(ir::Value value) {
    int& refcount = reference_count[value];
    if (refcount == 0) {
        refcount = value.references().size();
    }

    auto node = value.node();
    Condition_code cc;
    switch (node->opcode()) {
        case ir::Opcode::eq: cc = Condition_code::equal; break;
        case ir::Opcode::ne: cc = Condition_code::not_equal; break;
        case ir::Opcode::lt: cc = Condition_code::less; break;
        case ir::Opcode::ge: cc = Condition_code::greater_equal; break;
        case ir::Opcode::ltu: cc = Condition_code::below; break;
        case ir::Opcode::geu: cc = Condition_code::above_equal; break;
        default: ASSERT(0);
    }

    auto op0 = node->operand(0);
    auto op1 = node->operand(1);

    if (op0.is_const()) {
        std::swap(op0, op1);

        switch (cc) {
            case x86::Condition_code::less: cc = x86::Condition_code::greater; break;
            case x86::Condition_code::greater_equal: cc = x86::Condition_code::less_equal; break;
            case x86::Condition_code::below: cc = x86::Condition_code::above; break;
            case x86::Condition_code::above_equal: cc = x86::Condition_code::below_equal; break;
            default: break;
        }
    }

    ASSERT(!op0.is_const());

    // Decrease reference early, so if it is the last use we can eliminate one move.
    Register loc0 = get_register_location(op0);
    pin_register(loc0);
    emit(cmp(loc0, get_location_ex(op1, true, true)));
    unpin_register(loc0);

    if (--refcount == 0) {
        decrease_reference(op0);
        decrease_reference(op1);
    }

    return cc;
}

void Backend::after(ir::Node* node) {
    switch (node->opcode()) {
        case ir::Opcode::block:
        case ir::Opcode::jmp:
        case ir::Opcode::i_if:
            // These are not handled here.
            break;
        case ir::Opcode::constant:
            // constants are handled specially with in generation of each node that takes a value.
            break;
        case ir::Opcode::fence:
            // fence are only for ordering.
            break;
        case ir::Opcode::load_register: {
            auto output = node->value(1);
            uint16_t regnum = static_cast<ir::Register_access*>(node)->regnum();

            // Allocate and bind register.
            Register reg = alloc_register(output.type());
            bind_register(output, reg);

            // Move the emulated register to 64-bit version of allocated machine register.
            emit(mov(reg, qword(Register::rbp + regnum * 8)));
            break;
        }
        case ir::Opcode::store_register: {
            auto op = node->operand(1);
            uint16_t regnum = static_cast<ir::Register_access*>(node)->regnum();

            Operand loc = get_location_ex(op, false, true);
            decrease_reference(op);

            // Move the allocated machine register back to the emulated register.
            emit(mov(qword(Register::rbp + regnum * 8), loc));
            break;
        }
        case ir::Opcode::load_memory: {
            auto output = node->value(1);
            auto address = node->operand(1);

            if (emu::Flat_mmu* flat_mmu = dynamic_cast<emu::Flat_mmu*>(_state.mmu.get())) {
                Register reg = alloc_register(output.type());
                Register reg64 = modify_size(ir::Type::i64, reg).as_register();
                bind_register(output, reg);
                Operand mem_operand;

                if (address.is_const()) {
                    if (reg == Register::rax) {
                        mem_operand = reinterpret_cast<uintptr_t>(flat_mmu->memory_) + address.const_value();
                    } else {
                        emit(mov(reg64, reinterpret_cast<uintptr_t>(flat_mmu->memory_) + address.const_value()));
                        Memory mem = qword(reg64 + 0);
                        mem.size = get_type_size(output.type()) / 8;
                        mem_operand = mem;
                    }
                } else {
                    emit(mov(reg64, reinterpret_cast<uintptr_t>(flat_mmu->memory_)));
                    auto address_op = get_location(address);
                    Memory mem;
                    if (address_op.is_register()) {
                        mem = qword(reg64 + address_op.as_register() * 1);
                    } else {
                        emit(add(reg64, get_location(address)));
                        mem = qword(reg64 + 0);
                    }
                    mem.size = get_type_size(output.type()) / 8;
                    mem_operand = mem;
                    decrease_reference(address);
                }

                if (mem_operand.is_immediate()) {
                    emit(movabs(reg, mem_operand));
                } else {
                    emit(mov(reg, mem_operand));
                }
                break;
            }

            // Setup arguments.
            if (address.is_const()) {
                if (register_content[6]) spill_register(Register::rsi);
                emit(mov(Register::rsi, address.const_value()));
            } else {
                ensure_register(address, Register::rsi);
                decrease_reference(address);
            }

            // Store everything.
            spill_all_registers();

            // Setup other arguments and call.
            emit(mov(Register::rdi, reinterpret_cast<uintptr_t>(_state.mmu.get())));

            uintptr_t func;
            switch (output.type()) {
                case ir::Type::i8: func = reinterpret_cast<uintptr_t>(
                    AS_FUNCTION_POINTER(&emu::Paging_mmu::load_memory<uint8_t>)
                ); break;
                case ir::Type::i16: func = reinterpret_cast<uintptr_t>(
                    AS_FUNCTION_POINTER(&emu::Paging_mmu::load_memory<uint16_t>)
                ); break;
                case ir::Type::i32: func = reinterpret_cast<uintptr_t>(
                    AS_FUNCTION_POINTER(&emu::Paging_mmu::load_memory<uint32_t>)
                ); break;
                case ir::Type::i64: func = reinterpret_cast<uintptr_t>(
                    AS_FUNCTION_POINTER(&emu::Paging_mmu::load_memory<uint64_t>)
                ); break;
                default: ASSERT(0);
            }

            emit(mov(Register::rax, func));
            emit(call(Register::rax));

            bind_register(output, register_of_id(output.type(), 0));
            break;
        }
        case ir::Opcode::store_memory: {
            auto address = node->operand(1);
            auto value = node->operand(2);

            if (emu::Flat_mmu* flat_mmu = dynamic_cast<emu::Flat_mmu*>(_state.mmu.get())) {
                Register loc_value = Register::none;
                if (!value.is_const()) {
                    loc_value = get_register_location(value);
                    pin_register(loc_value);
                } else if (!is_int32(value.const_value())) {
                    loc_value = alloc_register(value.type());
                    emit(mov(loc_value, value.const_value()));
                    pin_register(loc_value);
                }

                Register loc = alloc_register(ir::Type::i64);
                Operand mem_operand;

                if (address.is_const()) {
                    // if (loc_value == 0) {
                    //     mem_operand = reinterpret_cast<uintptr_t>(flat_mmu->memory_) + address.const_value();
                    // } else {
                        emit(mov(loc, reinterpret_cast<uintptr_t>(flat_mmu->memory_) + address.const_value()));
                        Memory mem = qword(loc + 0);
                        mem.size = get_type_size(value.type()) / 8;
                        mem_operand = mem;
                    // }
                } else {
                    emit(mov(loc, reinterpret_cast<uintptr_t>(flat_mmu->memory_)));
                    auto address_op = get_location(address);
                    Memory mem;
                    if (address_op.is_register()) {
                        mem = qword(loc + address_op.as_register() * 1);
                    } else {
                        emit(add(loc, get_location(address)));
                        mem = qword(loc + 0);
                    }
                    mem.size = get_type_size(value.type()) / 8;
                    mem_operand = mem;
                    decrease_reference(address);
                }

                if (mem_operand.is_immediate()) {
                    emit(movabs(mem_operand, loc_value));
                } else {
                    if (loc_value == Register::none) {
                        emit(mov(mem_operand, value.const_value()));
                    } else {
                        emit(mov(mem_operand, loc_value));
                    }
                }

                if (loc_value != Register::none) {
                    unpin_register(loc_value);
                    decrease_reference(value);
                }

                break;
            }

            // Setup arguments.
            if (address.is_const()) {
                if (register_content[6]) spill_register(Register::rsi);
                emit(mov(Register::rsi, address.const_value()));
            } else {
                ensure_register(address, Register::rsi);
                decrease_reference(address);
            }

            if (value.is_const()) {
                if (register_content[2]) spill_register(Register::rdx);
                emit(mov(Register::rdx, value.const_value()));
            } else {
                ensure_register(value, register_of_id(value.type(), 2));
                decrease_reference(value);
            }

            // Store everything. Decrease reference before that to eliminate some redundant saves.
            spill_all_registers();

            // Setup other arguments and call.
            emit(mov(Register::rdi, reinterpret_cast<uintptr_t>(_state.mmu.get())));

            uintptr_t func;
            switch (value.type()) {
                case ir::Type::i8: func = reinterpret_cast<uintptr_t>(
                    AS_FUNCTION_POINTER(&emu::Paging_mmu::store_memory<uint8_t>)
                ); break;
                case ir::Type::i16: func = reinterpret_cast<uintptr_t>(
                    AS_FUNCTION_POINTER(&emu::Paging_mmu::store_memory<uint16_t>)
                ); break;
                case ir::Type::i32: func = reinterpret_cast<uintptr_t>(
                    AS_FUNCTION_POINTER(&emu::Paging_mmu::store_memory<uint32_t>)
                ); break;
                case ir::Type::i64: func = reinterpret_cast<uintptr_t>(
                    AS_FUNCTION_POINTER(&emu::Paging_mmu::store_memory<uint64_t>)
                ); break;
                default: ASSERT(0);
            }

            emit(mov(Register::rax, func));
            emit(call(Register::rax));
            break;
        }
        case ir::Opcode::emulate: {
            spill_all_registers();
            emit(mov(Register::rsi, node->attribute()));
            emit(mov(Register::rdi, Register::rbp));
            emit(mov(Register::rax, reinterpret_cast<uintptr_t>(riscv::step)));
            emit(call(Register::rax));
            break;
        }
        case ir::Opcode::cast: {
            auto output = node->value(0);
            auto op = node->operand(0);

            // Special handling for i1 upcast
            if (op.type() == ir::Type::i1) {
                Condition_code cc = emit_compare(op);

                // Allocate and bind register.
                Register reg = alloc_register(output.type());
                bind_register(output, reg);

                if (output.type() != ir::Type::i8) {
                    emit(mov(modify_size(ir::Type::i32, reg), 0));
                }
                emit(setcc(cc, modify_size(ir::Type::i8, reg)));
                break;
            }

            // Decrease reference early, so if it is the last use we can eliminate one move.
            Operand loc0 = get_location(op);
            decrease_reference(op);

            // Allocate and bind register. Try to use loc0 if possible to eliminate move.
            Register reg = alloc_register(output.type(), loc0);
            bind_register(output, reg);

            // Get size before and after the cast.
            auto op_type = op.type();
            int old_size = ir::get_type_size(op_type);
            int new_size = ir::get_type_size(output.type());

            if (old_size == 1) op_type = ir::Type::i8;

            if (old_size > new_size) {

                // Down-cast can be treated as simple move. If the size is less than 32-bit, we use 32-bit move.
                if (!same_location(loc0, reg)) {
                    emit_move(output.type(), reg, modify_size(output.type(), loc0));
                }

            } else {

                // Up-cast needs actual work.
                if (static_cast<ir::Cast*>(node)->sign_extend()) {
                    if (loc0.is_register()) {
                        Register loc0reg = loc0.as_register();
                        if (loc0reg == Register::eax && reg == Register::rax) {
                            emit(cdqe());
                            break;
                        }
                    }
                    emit(movsx(reg, loc0));
                } else {

                    // 32-bit to 64-bit cast is a move.
                    if (op_type == ir::Type::i32) {
                        emit(mov(modify_size(ir::Type::i32, reg), loc0));
                    } else {
                        emit(movzx(reg, loc0));
                    }
                }
            }

            break;
        }
        case ir::Opcode::mux: {

            auto output = node->value(0);
            auto op0 = node->operand(0);
            auto op1 = node->operand(1);
            auto op2 = node->operand(2);

            Condition_code cc = emit_compare(op0);

            // Decrease reference early, so if it is the last use we can eliminate one move.
            Operand loc2 = Register::none;
            if (!op2.is_const()) {
                loc2 = get_location(op2);
                decrease_reference(op2);
            }

            // Allocate and bind register. Try to use loc2 if possible to eliminate move.
            Register reg = alloc_register(output.type(), loc2);
            bind_register(output, reg);

            // Move false operand to register.
            if ((loc2.is_register() && loc2.as_register() == Register::none) || !same_location(loc2, reg)) {
                if (op2.is_const()) {
                    emit(mov(reg, op2.const_value()));
                } else {
                    emit_move(output.type(), reg, loc2);
                }
            }

            pin_register(reg);
            emit(cmovcc(cc, reg, get_location_ex(op1, true, false)));
            unpin_register(reg);
            decrease_reference(op1);
            break;
        }
        case ir::Opcode::add: emit_alu(node, Opcode::add); break;
        case ir::Opcode::sub: emit_alu(node, Opcode::sub); break;
        case ir::Opcode::i_xor: emit_alu(node, Opcode::i_xor); break;
        case ir::Opcode::i_and: emit_alu(node, Opcode::i_and); break;
        case ir::Opcode::i_or: emit_alu(node, Opcode::i_or); break;
        case ir::Opcode::shl: emit_shift(node, Opcode::shl); break;
        case ir::Opcode::shr: emit_shift(node, Opcode::shr); break;
        case ir::Opcode::sar: emit_shift(node, Opcode::sar); break;
        case ir::Opcode::i_not: emit_unary(node, Opcode::i_not); break;
        case ir::Opcode::neg: emit_unary(node, Opcode::neg); break;
        // i1 nodes are handled by their users.
        case ir::Opcode::eq:
        case ir::Opcode::ne:
        case ir::Opcode::lt:
        case ir::Opcode::ge:
        case ir::Opcode::ltu:
        case ir::Opcode::geu: break;
        default: ASSERT(0);
    }
}

// Clear all code generation information. These are relevant only in a basic block.
void Backend::clear() {
    stack_size = 0;
    reference_count.clear();
    location.clear();
    memory_location.clear();
    for (auto& loc: register_content) loc = {};
    for (auto& loc: pinned) loc = false;
}

void Backend::run(ir::Graph& graph) {
    std::vector<ir::Node*> blocks;
    std::vector<ir::Node*> to_process;

    auto start = graph.start();
    ASSERT(start->value(0).references().size() == 1);
    to_process.push_back(*start->value(0).references().begin());

    // Some very naive basic block scheduling.
    while (!to_process.empty()) {
        auto block = to_process.back();
        to_process.pop_back();

        if (block->opcode() == ir::Opcode::end) continue;

        // Make sure the block is not yet in the list.
        if (std::find(blocks.begin(), blocks.end(), block) != blocks.end()) {
            continue;
        }

        blocks.push_back(block);

        ASSERT(block->opcode() == ir::Opcode::block);

        auto end = static_cast<ir::Block*>(block)->end();

        if (end->opcode() == ir::Opcode::i_if) {
            to_process.push_back(*end->value(0).references().begin());
            to_process.push_back(*end->value(1).references().begin());
        } else {
            ASSERT(end->opcode() == ir::Opcode::jmp);
            to_process.push_back(*end->value(0).references().begin());
        }
    }

    // Now we have a linear list of blocks.

    // Generate epilogue.
    emit(push(Register::rbp));
    emit(mov(Register::rbp, Register::rdi));

    // In the simple case, we don't have to do relocations.
    if (blocks.size() == 1) {
        run_on(graph, static_cast<ir::Block*>(blocks[0])->end());
        emit(pop(Register::rbp));
        emit(ret());
        return;
    }

    // Push end to the block list to ease processing.
    blocks.push_back(graph.root());

    // These are used for relocation
    std::unordered_map<ir::Node*, size_t> label_def;
    std::unordered_map<ir::Node*, std::vector<size_t>> label_use;

    for (size_t i = 0; i < blocks.size() - 1; i++) {
        auto block = blocks[i];
        auto next_block = blocks[i + 1];
        auto end = static_cast<ir::Block*>(block)->end();

        // Store the label for relocation purpose.
        label_def[block] = _encoder.buffer().size();

        // Generate code for the block.
        run_on(graph, end);

        // This records the fallthrough target.
        ir::Node* target = nullptr;

        if (end->opcode() == ir::Opcode::i_if) {

            // Extract targets
            auto true_target = *end->value(0).references().begin();
            target = *end->value(1).references().begin();

            auto op = end->operand(1);
            if (op.is_const()) {
                if (op.const_value()) target = true_target;
            } else {
                Condition_code cc = emit_compare(op);
                if (true_target == next_block) {
                    // Invert condition code.
                    cc = static_cast<Condition_code>(static_cast<uint8_t>(cc) ^ 1);
                    std::swap(true_target, target);
                }
                emit(jcc(cc, 0xCAFE));
                label_use[true_target].push_back(_encoder.buffer().size());
            }
        } else {
            ASSERT(end->opcode() == ir::Opcode::jmp);
            target = *end->value(0).references().begin();
        }

        if (target != next_block) {
            emit(jmp(0xBEEF));
            label_use[target].push_back(_encoder.buffer().size());
        }

        clear();
    }

    label_def[graph.root()] = _encoder.buffer().size();

    // Patching labels
    for (const auto& pair: label_def) {
        auto& uses = label_use[pair.first];
        for (auto use: uses) {
            util::write_as<uint32_t>(_encoder.buffer().data() + use - 4, pair.second - use);
        }
    }

    emit(pop(Register::rbp));
    emit(ret());
}

}
