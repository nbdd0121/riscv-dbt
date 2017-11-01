#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "emu/mmu.h"
#include "emu/state.h"
#include "riscv/context.h"
#include "riscv/decoder.h"
#include "riscv/disassembler.h"
#include "riscv/instruction.h"
#include "util/format.h"

namespace emu {
reg_t load_elf(const char *filename, State& mmu);
}

namespace riscv {
void step(Context *context, Instruction inst);
}

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
        } else {
            std::cerr << "Unknown argument " << arg << ", ignored" << std::endl;
        }
    }

    // The next argument is the path to the executable.
    if (arg_index == argc) {
        std::cerr << "Program name unspecified" << std::endl;
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

    try {
        while (state.exit_code == -1) {
            decoder.pc(context->pc);
            riscv::Instruction inst = decoder.decode_instruction();

            // Disassembler::print_instruction(context->pc, inst_bits, inst);
            context->pc += inst.length();
            riscv::step(context, inst);
            context->instret++;
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
