#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "emu/mmu.h"
#include "riscv/decoder.h"
#include "riscv/disassembler.h"
#include "riscv/instruction.h"

namespace emu {
reg_t load_elf(const char *filename, Mmu& mmu);
}

int main(int argc, const char **argv) {

    /* Arguments to be parsed */
    // By default we use flat mmu since it is faster.
    bool use_paging = false;

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

    // Before we setup argv and envp passed to the emulated program, we need to get the MMU functional first.
    std::unique_ptr<emu::Mmu> mmu;
    if (use_paging) {
        mmu = std::make_unique<emu::Paging_mmu>();
    } else {
        mmu = std::make_unique<emu::Flat_mmu>(0x10000000);
    }

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

    emu::reg_t pc = load_elf(program_name, *mmu);
    for (int i = 0; i < 100; i++) {
        uint32_t inst_bits = mmu->load_memory<uint32_t>(pc);
        riscv::Decoder decoder { inst_bits };
        riscv::Instruction inst = decoder.decode();
        riscv::Disassembler::print_instruction(pc, inst_bits, inst);
        pc += inst.length();
    }
}
