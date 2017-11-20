#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "emu/mmu.h"
#include "emu/state.h"
#include "main/dbt.h"
#include "main/interpreter.h"
#include "main/signal.h"
#include "riscv/basic_block.h"
#include "riscv/context.h"
#include "riscv/decoder.h"
#include "riscv/disassembler.h"
#include "riscv/instruction.h"
#include "riscv/opcode.h"
#include "util/format.h"

static const char *usage_string = "Usage: {} [options] program [arguments...]\n\
Options:\n\
  --paging      Use soft paging MMU instead of a flat MMU. The emulated program\n\
                will have larger address space in expense of performance.\n\
  --strace      Log system calls.\n\
  --disassemble Log decoded instructions.\n\
  --disable-dbt Disable dynamic binary translation and use interpretation instead.\n\
  --help        Display this help message.\n\
";

int main(int argc, const char **argv) {

    setup_fault_handler();

    /* Arguments to be parsed */
    // By default we use flat mmu since it is faster.
    bool use_paging = false;
    bool strace = false;
    bool disassemble = false;
    bool use_dbt = true;

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
        } else if (strcmp(arg, "--disable-dbt") == 0) {
            use_dbt = false;
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

    // Before we setup argv and envp passed to the emulated program, we need to get the MMU functional first.
    if (use_paging) {
        state.mmu = std::make_unique<emu::Paging_mmu>();
    } else {
        state.mmu = std::make_unique<emu::Flat_mmu>(0x10000000);
    }

    emu::Mmu *mmu = state.mmu.get();

    // Set sp to be the highest possible address.
    emu::reg_t sp = use_paging ? 0x800000000000 : 0x10000000;
    mmu->allocate_page(sp - 0x800000, 0x800000);

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

    try {
        if (use_dbt) {
            Dbt_runtime executor { state };
            while (true) {
                executor.step(*context);
            }
        } else {
            Interpreter executor { state };
            while (true) {
                executor.step(*context);
            }
        }
    } catch (emu::Exit_control& ex) {
        return ex.exit_code;
    } catch (std::exception& ex) {
        util::print("{}\npc={:x}\n", ex.what(), context->pc);
        for (int i = 0; i < 32; i++) {
            util::print("x{} = {:x}\n", i, context->registers[i]);
        }
        return 1;
    }
}
