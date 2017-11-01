#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "emu/mmu.h"
#include "emu/state.h"
#include "riscv/basic_block.h"
#include "riscv/context.h"
#include "riscv/decoder.h"
#include "riscv/disassembler.h"
#include "riscv/instruction.h"
#include "riscv/opcode.h"
#include "util/format.h"

namespace emu {
reg_t load_elf(const char *filename, State& mmu);
}

namespace riscv {
void step(Context *context, Instruction inst);
}

static const char *usage_string = "Usage: {} [options] program [arguments...]\n\
Options:\n\
  --paging      Use soft paging MMU instead of a flat MMU. The emulated program\n\
                will have larger address space in expense of performance.\n\
  --strace      Log system calls.\n\
  --disassemble Log decoded instructions.\n\
  --help        Display this help message.\n\
";

int main(int argc, const char **argv) {

    /* Arguments to be parsed */
    // By default we use flat mmu since it is faster.
    bool use_paging = false;
    bool strace = false;
    bool disassemble = false;

    // Parsing arguments
    int arg_index;
    for (arg_index = 1; arg_index < argc; arg_index++) {
        const char *arg = argv[arg_index];

        // We've parsed all arguments. This indicates the name of the executable.
        if (arg[0] != '-') {
            break;
        }

        if (strcmp(arg, "--paging") == 0) {
            use_paging = true;
        } else if (strcmp(arg, "--strace") == 0) {
            strace = true;
        } else if (strcmp(arg, "--disassemble") == 0) {
            disassemble = true;
        } else if (strcmp(arg, "--help") == 0) {
            util::error(usage_string, argv[0]);
            return 0;
        } else {
            util::error("{}: unrecognized option '{}'\n", argv[0], arg);
        }
    }

    // The next argument is the path to the executable.
    if (arg_index == argc) {
        util::error(usage_string, argv[0]);
        return 1;
    }
    const char *program_name = argv[arg_index];

    emu::State state;
    state.strace = strace;
    state.disassemble = disassemble;
    state.exit_code = -1;

    // Before we setup argv and envp passed to the emulated program, we need to get the MMU functional first.
    if (use_paging) {
        state.mmu = std::make_unique<emu::Paging_mmu>();
    } else {
        state.mmu = std::make_unique<emu::Flat_mmu>(0x10000000);
    }

    emu::Mmu *mmu = state.mmu.get();

    // Set sp to be the highest possible address.
    emu::reg_t sp = use_paging ? 0x800000000000 : 0x10000000;

    // This contains (guest) pointers to all argument strings.
    std::vector<emu::reg_t> arg_pointers(argc - arg_index);

    // Copy all arguments into guest user space.
    for (int i = argc - 1; i >= arg_index; i--) {
        size_t arg_length = strlen(argv[i]) + 1;

        // Allocate memory from stack and copy to that region.
        sp -= arg_length;
        mmu->copy_from_host(sp, argv[i], arg_length);
        arg_pointers[i - arg_index] = sp;
    }

    // Align the stack to 8-byte boundary.
    sp &= ~7;

    // AT_NULL = 0
    sp -= 2 * sizeof(emu::reg_t);

    // envp = 0
    sp -= sizeof(emu::reg_t);

    // argv[argc] = 0
    sp -= sizeof(emu::reg_t);

    // fill in argv
    sp -= (argc - arg_index) * sizeof(emu::reg_t);
    mmu->copy_from_host(sp, arg_pointers.data(), (argc - arg_index) * sizeof(emu::reg_t));

    // set argc
    sp -= sizeof(emu::reg_t);
    mmu->store_memory<emu::reg_t>(sp, argc - arg_index);

    state.context = std::make_unique<riscv::Context>();
    riscv::Context *context = state.context.get();

    // Initialize context
    context->pc = load_elf(program_name, state);

    for (int i = 1; i < 32; i++) {
        // Reset to some easily debuggable value.
        context->registers[i] = 0xCCCCCCCCCCCCCCCC;
        context->fp_registers[i] = 0xFFFFFFFFFFFFFFFF;
    }

    context->mmu = mmu;
    context->state = &state;

    // x0 must always be 0
    context->registers[0] = 0;
    // sp
    context->registers[2] = sp;
    context->fcsr = 0;
    context->instret = 0;
    context->lr = 0;

    riscv::Decoder decoder { &state };
    std::unordered_map<emu::reg_t, riscv::Basic_block> inst_cache;

    try {
        while (state.exit_code == -1) {
            emu::reg_t pc = context->pc;
            riscv::Basic_block& basic_block = inst_cache[pc];

            if (basic_block.instructions.size() == 0) {
                decoder.pc(pc);
                basic_block = decoder.decode_basic_block();

                // Post-processing by replacing auipc with lui.
                // This is a little bit trick: we rely on the fact that handling of immediate format is done in the decoder
                // instead of in the interpreter, so lui is implemented as write_rd(imm)
                for (auto& inst: basic_block.instructions) {
                    if (inst.opcode() == riscv::Opcode::auipc) {
                        inst.opcode(riscv::Opcode::lui);
                        inst.imm(inst.imm() + pc);
                    }
                    pc += inst.length();
                }
            }

            size_t block_size = basic_block.instructions.size() - 1;

            for (size_t i = 0; i < block_size; i++) {
                // Retrieve cached data
                riscv::Instruction inst = basic_block.instructions[i];
                riscv::step(context, inst);
            }

            context->pc = basic_block.end_pc;
            context->instret += block_size + 1;
            riscv::Instruction inst = basic_block.instructions[block_size];
            if (inst.opcode() == riscv::Opcode::fence_i) {
                inst_cache.clear();
            } else {
                riscv::step(context, inst);
            }
        }
    } catch (std::exception& ex) {
        util::print("{}\npc={:x}\n", ex.what(), context->pc);
        for (int i = 0; i < 32; i++) {
            util::print("x{} = {:x}\n", i, context->registers[i]);
        }
        return 1;
    }

    return state.exit_code;
}
