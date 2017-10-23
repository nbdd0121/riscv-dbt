#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

#include <stdexcept>

#include "emu/elf64.h"
#include "emu/mmu.h"

#include "util/scope_exit.h"

namespace emu {

reg_t load_elf(const char *filename, Mmu& mmu) {

    // Open the file first
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error { "cannot open file" };
    }

    // Make sure the file descriptor does not leak when this function returns.
    SCOPE_EXIT {
        close(fd);
    };

    // Get the size of the file.
    struct stat s;
    int status = fstat(fd, &s);
    if (status == -1) {
        throw std::runtime_error { "cannot fstat file" };
    }
    
    // Map the file to memory.
    std::byte *memory = reinterpret_cast<std::byte*>(mmap(nullptr, s.st_size, PROT_READ, MAP_PRIVATE, fd, 0));
    if (memory == nullptr) {
        throw std::runtime_error { "cannot mmap file" };
    }

    // Make sure mmap does not leak.
    SCOPE_EXIT {
        munmap(memory, s.st_size);
    };

    // Parse the ELF header and load the binary into memory.
    elf::Elf64_Ehdr *header = reinterpret_cast<elf::Elf64_Ehdr*>(memory);
    for (int i = 0; i < header->e_phnum; i++) {
        elf::Elf64_Phdr *h = reinterpret_cast<elf::Elf64_Phdr*>(memory + header->e_phoff + header->e_phentsize * i);
        if (h->p_type == 1) {
            // MMU should have memory zeroes at startup
            mmu.zero_memory(h->p_vaddr + h->p_filesz, h->p_memsz - h->p_filesz);
            mmu.copy_from_host(h->p_vaddr, memory + h->p_offset, h->p_filesz);
        }
    }

    return header->e_entry;
}

}
