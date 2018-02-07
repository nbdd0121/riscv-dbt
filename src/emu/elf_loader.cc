#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>

#include "emu/elf64.h"
#include "emu/mmu.h"
#include "emu/state.h"

#include "util/scope_exit.h"

namespace emu {

struct Elf_file {
    int fd = -1;
    long file_size;
    std::byte *memory = nullptr;

    ~Elf_file();

    void load(const char* filename);
    void validate();
    std::string find_interpreter();
};

Elf_file::~Elf_file() {
    if (fd != -1) {
        close(fd);
    }

    if (memory == nullptr) {
        munmap(memory, file_size);
    }
}

void Elf_file::load(const char *filename) {

    // Open the file first
    fd = open(filename, O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error { "cannot open file" };
    }

    // Get the size of the file.
    {
        struct stat s;
        int status = fstat(fd, &s);
        if (status == -1) {
            throw std::runtime_error { "cannot fstat file" };
        }

        file_size = s.st_size;
    }

    // Map the file to memory.
    memory = reinterpret_cast<std::byte*>(mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (memory == nullptr) {
        throw std::runtime_error { "cannot mmap file" };
    }
}

void Elf_file::validate() {
    elf::Elf64_Ehdr *header = reinterpret_cast<elf::Elf64_Ehdr*>(memory);

    // Check the ELF magic numbers
    if (memcmp(header->e_ident, "\177ELF", 4) != 0) {
        throw std::runtime_error { "the program to be loaded is not elf." };
    }

    // We can only proceed with executable or dynamic binary.
    if (header->e_type != elf::ET_EXEC && header->e_type != elf::ET_DYN) {
        throw std::runtime_error { "the binary is not executable." };
    }

    // Check that the ELF is for RISC-V
    if (header->e_machine != elf::EM_RISCV) {
        throw std::runtime_error { "the binary is not for RISC-V." };
    }

    // TODO: Also check for flags about ISA extensions
}

std::string Elf_file::find_interpreter() {
    elf::Elf64_Ehdr *header = reinterpret_cast<elf::Elf64_Ehdr*>(memory);
    for (int i = 0; i < header->e_phnum; i++) {
        elf::Elf64_Phdr *h = reinterpret_cast<elf::Elf64_Phdr*>(memory + header->e_phoff + header->e_phentsize * i);
        if (h->p_type == elf::PT_INTERP) {
            if (*reinterpret_cast<char*>(memory + h->p_offset + h->p_filesz - 1) != 0) {
                throw std::runtime_error { "interpreter name should be null-terminated." };
            }

            return reinterpret_cast<char*>(memory + h->p_offset);
        }
    }

    return {};
}

reg_t load_elf(const char *filename, State& state) {

    Elf_file file;
    file.load(filename);
    file.validate();

    Mmu& mmu = *state.mmu;

    // Parse the ELF header and load the binary into memory.
    auto memory = file.memory;
    elf::Elf64_Ehdr *header = reinterpret_cast<elf::Elf64_Ehdr*>(memory);

    reg_t brk = 0;
    for (int i = 0; i < header->e_phnum; i++) {
        elf::Elf64_Phdr *h = reinterpret_cast<elf::Elf64_Phdr*>(memory + header->e_phoff + header->e_phentsize * i);
        if (h->p_type == elf::PT_LOAD) {

            // size in memory cannot be smaller than size in file
            if (h->p_filesz > h->p_memsz) {
                throw std::runtime_error { "invalid elf file: constraint p_filesz <= p_memsz is not satisified" };
            }

            reg_t vaddr_end = h->p_vaddr + h->p_memsz;
            reg_t page_start = h->p_vaddr &~ page_mask;
            reg_t page_end = (vaddr_end + page_mask) &~ page_mask;
            mmu.allocate_page(page_start, page_end - page_start);

            // MMU should have memory zeroes at startup
            mmu.zero_memory(h->p_vaddr + h->p_filesz, h->p_memsz - h->p_filesz);
            // TODO: This should be be mmapped instead of copied
            mmu.copy_from_host(h->p_vaddr, memory + h->p_offset, h->p_filesz);

            // Set brk to the address past the last program segment.
            if (vaddr_end > brk) {
                brk = vaddr_end;
            }
        }
    }

    state.original_brk = brk;
    state.brk = brk;
    state.heap_start = (brk + page_mask) &~ page_mask;
    state.heap_end = state.heap_start;

    std::string interpreter = file.find_interpreter();
    if (!interpreter.empty()) {
        throw std::runtime_error { "sad. dynamic binaries are not yet supported." };
    }

    return header->e_entry;
}

}
