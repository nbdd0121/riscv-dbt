#include <cstring>
#include <sys/mman.h>

#include "emu/mmu.h"
#include "util/memory.h"

namespace emu {

Flat_mmu::Flat_mmu(size_t size): size_(size) {

    // Ideally we want the page to be all PROT_NONE, then allocate when it is needed.
    memory_ = reinterpret_cast<std::byte*>(
        mmap(NULL, size, PROT_NONE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)
    );

    if (memory_ == nullptr) {
        throw std::bad_alloc {};
    }
}

Flat_mmu::~Flat_mmu() {
    munmap(memory_, size_);
}

// Allocate memory at given virtual address and estabilish mapping.
// The address and size must be page-aligned.
void Flat_mmu::allocate_page(reg_t address, reg_t size) {

    // The input must be page-aligned.
    ASSERT((address & page_mask) == 0 && (size & page_mask) == 0);

    if (address >= size_) {
        throw std::bad_alloc {};
    }

    if (mprotect(memory_ + address, size, PROT_READ | PROT_WRITE) == -1) {
        throw std::bad_alloc {};
    }
}

// Allocate memory at given virtual address and estabilish mapping.
// The address and size must be page-aligned.
void Id_mmu::allocate_page(reg_t address, reg_t size) {

    // The input must be page-aligned.
    ASSERT((address & page_mask) == 0 && (size & page_mask) == 0);

    auto ptr = translate(address);

    void *ret = mmap(ptr, size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (ret != ptr) {
        throw std::bad_alloc {};
    }

    return;
}

}
