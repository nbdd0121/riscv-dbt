#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "emu/mmu.h"
#include "emu/state.h"
#include "main/dbt.h"
#include "main/interpreter.h"
#include "main/ir_dbt.h"
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
  --no-direct-memory    Disable generation of memory access instruction, use\n\
                        call to helper function instead.\n\
  --strace              Log system calls.\n\
  --disassemble         Log decoded instructions.\n\
  --engine=interpreter  Use interpreter instead of dynamic binary translator.\n\
  --engine=dbt          Use simple binary translator instead of IR-based\n\
                        optimising binary translator.\n\
  --with-instret        Enable precise instret updating in binary translated\n\
                        code.\n\
  --strict-exception    Enable strict enforcement of excecution correctness in\n\
                        case of segmentation fault.\n\
  --inline-limit=<n>    Number of basic blocks that can be inlined in single\n\
                        compilation by the IR-based binary translator.\n\
  --monitor-performance Display metrics about performance in compilation phase.\n\
  --help                Display this help message.\n\
";

int main(int argc, const char **argv) {

    setup_fault_handler();

    /* Arguments to be parsed */
    bool use_dbt = false;
    bool use_ir = true;

    emu::State state;
    state.strace = false;
    state.disassemble = false;
    state.no_instret = true;
    state.inline_limit = 16;

    // Parsing arguments
    int arg_index;
    for (arg_index = 1; arg_index < argc; arg_index++) {
        const char *arg = argv[arg_index];

        // We've parsed all arguments. This indicates the name of the executable.
        if (arg[0] != '-') {
            break;
        }

        if (strcmp(arg, "--no-direct-memory") == 0) {
            emu::no_direct_memory_access = true;
        } else if (strcmp(arg, "--strace") == 0) {
            state.strace = true;
        } else if (strcmp(arg, "--disassemble") == 0) {
            state.disassemble = true;
        } else if (strcmp(arg, "--engine=dbt") == 0) {
            use_ir = false;
            use_dbt = true;
        } else if (strcmp(arg, "--engine=interpreter") == 0) {
            use_ir = false;
            use_dbt = false;
        } else if (strcmp(arg, "--with-instret") == 0) {
            state.no_instret = false;
        } else if (strcmp(arg, "--strict-exception") == 0) {
            emu::strict_exception = true;
        } else if (strncmp(arg, "--inline-limit=", strlen("--inline-limit=")) == 0) {
            state.inline_limit = atoi(arg + strlen("--inline-limit="));
        } else if (strcmp(arg, "--monitor-performance") == 0) {
            emu::monitor_performance = true;
        } else if (strcmp(arg, "--help") == 0) {
            util::error(usage_string, argv[0]);
            return 0;
        } else {
            util::error("{}: unrecognized option '{}'\n", argv[0], arg);
            return 1;
        }
    }

    // The next argument is the path to the executable.
    if (arg_index == argc) {
        util::error(usage_string, argv[0]);
        return 1;
    }
    const char *program_name = argv[arg_index];

    // Set sp to be the highest possible address.
    emu::reg_t sp = 0x7fff00000000;
    emu::allocate_page(sp - 0x800000, 0x800000);

    // This contains (guest) pointers to all argument strings.
    std::vector<emu::reg_t> arg_pointers(argc - arg_index);

    // Copy all arguments into guest user space.
    for (int i = argc - 1; i >= arg_index; i--) {
        size_t arg_length = strlen(argv[i]) + 1;

        // Allocate memory from stack and copy to that region.
        sp -= arg_length;
        emu::copy_from_host(sp, argv[i], arg_length);
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
    emu::copy_from_host(sp, arg_pointers.data(), (argc - arg_index) * sizeof(emu::reg_t));

    // set argc
    sp -= sizeof(emu::reg_t);
    emu::store_memory<emu::reg_t>(sp, argc - arg_index);

    state.context = std::make_unique<riscv::Context>();
    riscv::Context *context = state.context.get();

    // Initialize context
    context->pc = load_elf(program_name, state);

    for (int i = 1; i < 32; i++) {
        // Reset to some easily debuggable value.
        context->registers[i] = 0xCCCCCCCCCCCCCCCC;
        context->fp_registers[i] = 0xFFFFFFFFFFFFFFFF;
    }

    context->state = &state;

    // x0 must always be 0
    context->registers[0] = 0;
    // sp
    context->registers[2] = sp;
    context->fcsr = 0;
    context->instret = 0;
    context->lr = 0;

    try {
        if (use_ir) {
            Ir_dbt executor { state };
            context->executor = &executor;
            while (true) {
                executor.step(*context);
            }
        } else if (use_dbt) {
            Dbt_runtime executor { state };
            context->executor = &executor;
            while (true) {
                executor.step(*context);
            }
        } else {
            Interpreter executor { state };
            context->executor = &executor;
            while (true) {
                executor.step(*context);
            }
        }
    } catch (emu::Exit_control& ex) {
        return ex.exit_code;
    } catch (std::exception& ex) {
        util::print("{}\npc  = {:16x}  ra  = {:16x}\n", ex.what(), context->pc, context->registers[1]);
        for (int i = 2; i < 32; i += 2) {
            util::print(
                "{:-3} = {:16x}  {:-3} = {:16x}\n",
                riscv::Disassembler::register_name(i), context->registers[i],
                riscv::Disassembler::register_name(i + 1), context->registers[i + 1]
            );
        }
        return 1;
    }
}
