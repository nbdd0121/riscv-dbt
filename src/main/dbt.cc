#include "emu/state.h"
#include "main/dbt.h"
#include "riscv/basic_block.h"
#include "riscv/context.h"
#include "riscv/decoder.h"
#include "riscv/instruction.h"
#include "riscv/opcode.h"
#include "util/assert.h"
#include "util/code_buffer.h"
#include "util/format.h"
#include "util/memory.h"
#include "x86/builder.h"
#include "x86/disassembler.h"
#include "x86/encoder.h"
#include "x86/instruction.h"
#include "x86/opcode.h"

// Shorthand for instruction coding.
using namespace x86::builder;

// A separate class is used instead of generating code directly in Dbt_runtime, so it is easier to define and use
// helper functions that are shared by many instructions.
class Dbt_compiler {
private:
    Dbt_runtime& runtime_;
    x86::Encoder encoder_;

    Dbt_compiler& operator <<(const x86::Instruction& inst);

public:
    Dbt_compiler(Dbt_runtime& runtime, util::Code_buffer& buffer): runtime_{runtime}, encoder_{buffer} {}
    void compile(emu::reg_t pc);
};

Dbt_runtime::Dbt_runtime(emu::State& state): state_ {state} {
    icache_tag_ = std::unique_ptr<emu::reg_t[]> { new emu::reg_t[4096] };
    icache_ = std::unique_ptr<std::byte*[]> { new std::byte*[4096] };
}

void Dbt_runtime::step(riscv::Context& context) {
    const emu::reg_t pc = context.pc;
    const ptrdiff_t tag = (pc >> 1) & 4095;

    // If the cache misses, compile the current block.
    if (UNLIKELY(icache_tag_[tag] != pc)) {
        compile(pc);
    }

    auto func = reinterpret_cast<void(*)(riscv::Context&)>(icache_[tag]);
    ASSERT(func);
    func(context);
    return;
}

void Dbt_runtime::compile(emu::reg_t pc) {
    const ptrdiff_t tag = (pc >> 1) & 4095;
    util::Code_buffer& buffer = inst_cache_[pc];

    // Reserve a page in case that the buffer is empty, it saves the code buffer from reallocating (which is expensive
    // as code buffer is backed up by mmap and munmap at the moment.
    // If buffer.size() is not zero, it means that we have compiled the code previously but it is not in the hot cache.
    if (buffer.size() == 0) {
        buffer.reserve(4096);
        Dbt_compiler compiler { *this, buffer };
        compiler.compile(pc);
    }

    // Update tag to reflect newly compiled code.
    icache_[tag] = buffer.data();
    icache_tag_[tag] = pc;
}

Dbt_compiler& Dbt_compiler::operator <<(const x86::Instruction& inst) {
    bool disassemble = runtime_.state_.disassemble;
    std::byte *pc;
    if (disassemble) {
        pc = encoder_.buffer().data() + encoder_.buffer().size();
    }
    encoder_.encode(inst);
    if (disassemble) {
        std::byte *new_pc = encoder_.buffer().data() + encoder_.buffer().size();
        x86::disassembler::print_instruction(
            reinterpret_cast<uintptr_t>(pc), reinterpret_cast<const char*>(pc), new_pc - pc, inst);
    }
    return *this;
}

#define memory_of_register(reg) (x86::Register::rbp + (offsetof(riscv::Context, registers) + sizeof(emu::reg_t) * reg - 0x80))
#define memory_of(name) (x86::Register::rbp + (offsetof(riscv::Context, name) - 0x80))

void Dbt_compiler::compile(emu::reg_t pc) {
    riscv::Decoder decoder { &runtime_.state_, pc };
    riscv::Basic_block block = decoder.decode_basic_block();

    if (runtime_.state_.disassemble) {
        util::log("Translating {:x} to {:x}\n", pc, reinterpret_cast<uintptr_t>(encoder_.buffer().data()));
    }

    // Prolog. We place context + 0x80 to rbp instead of context directly, as it allows all registers to be located
    // within int8 offset from rbp, so the assembly representation will uses a shorter encoding.
    *this << push(x86::Register::rbp);
    *this << lea(x86::Register::rbp, qword(x86::Register::rdi + 0x80));

    int pc_diff = 0;
    int instret_diff = 0;

    // We treat the last instruction differently.
    for (size_t i = 0; i < block.instructions.size() - 1; i++) {

        riscv::Instruction inst = block.instructions[i];
        riscv::Opcode opcode = inst.opcode();
        const int rd = inst.rd();

        switch (opcode) {
            case riscv::Opcode::auipc: {
                if (rd == 0) break;
                *this << mov(x86::Register::rax, qword(memory_of(pc)));
                *this << add(x86::Register::rax, pc_diff + inst.imm());
                *this << mov(qword(memory_of_register(rd)), x86::Register::rax);
                break;
            }
            default:
                *this << mov(x86::Register::rsi, util::read_as<uint64_t>(&inst));
                *this << lea(x86::Register::rdi, qword(x86::Register::rbp - 0x80));
                *this << mov(x86::Register::rax, reinterpret_cast<uintptr_t>(riscv::step));
                *this << call(x86::Register::rax);
                break;
        }

        pc_diff += inst.length();
        instret_diff++;
    }


    riscv::Instruction inst = block.instructions.back();

    *this << add(qword(memory_of(pc)), pc_diff + inst.length());
    *this << add(qword(memory_of(instret)), instret_diff + 1);

    if (inst.opcode() == riscv::Opcode::fence_i) {
        void (*callback)(Dbt_runtime&) = [](Dbt_runtime& runtime) {
            for (int i = 0; i < 4096; i++)
                runtime.icache_tag_[i] = 0;
            runtime.inst_cache_.clear();
        };
        *this << mov(x86::Register::rdi, reinterpret_cast<uintptr_t>(&runtime_));
        *this << mov(x86::Register::rax, reinterpret_cast<uintptr_t>(callback));
        *this << pop(x86::Register::rbp);
        *this << jmp(x86::Register::rax);
    }

    *this << mov(x86::Register::rsi, util::read_as<uint64_t>(&inst));
    *this << lea(x86::Register::rdi, qword(x86::Register::rbp - 0x80));
    *this << mov(x86::Register::rax, reinterpret_cast<uintptr_t>(riscv::step));
    *this << pop(x86::Register::rbp);
    *this << jmp(x86::Register::rax);
}
