#include <cstring>
#include <sys/mman.h>

#include "emu/mmu.h"
#include "util/memory.h"

namespace emu {

// Allocate memory at given virtual address and estabilish mapping.
// The address and size must be page-aligned.
void allocate_page(reg_t address, reg_t size) {

    // The input must be page-aligned.
    ASSERT((address & page_mask) == 0 && (size & page_mask) == 0);

    auto ptr = translate_address(address);

    void *ret = mmap(ptr, size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (ret != ptr) {
        throw std::bad_alloc {};
    }

    return;
}

}
