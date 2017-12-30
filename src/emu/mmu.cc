#include <cstring>
#include <sys/mman.h>

#include "emu/mmu.h"
#include "util/memory.h"

namespace emu {

template<typename T>
T Mmu::load_memory_misaligned(reg_t address) {
    reg_t page_offset = address & page_mask;
    reg_t page_base = address &~ page_mask;
    std::byte *translated_page = translate_page(page_base);

    if (UNLIKELY(page_offset > page_size - sizeof(T))) {

        // Unaligned, cross-page access. Slow path
        std::byte word[sizeof(T)];

        std::byte *next_translated_page = translate_page(page_base + page_size);
        int size_in_page = page_size - page_offset;
        util::safe_memcpy(word, translated_page + page_offset, size_in_page);
        util::safe_memcpy(word + size_in_page, next_translated_page, sizeof(T) - size_in_page);

        return util::read_as<T>(word);
    }

    return util::safe_read<T>(translated_page + page_offset);
}

template<typename T>
void Mmu::store_memory_misaligned(reg_t address, T value) {
    reg_t page_offset = address & page_mask;
    reg_t page_base = address &~ page_mask;
    std::byte *translated_page = translate_page(page_base);

    if (UNLIKELY(page_offset > page_size - sizeof(T))) {

        // Unaligned, cross-page access. Slow path
        std::byte word[sizeof(T)];
        util::write_as<T>(word, value);

        std::byte *next_translated_page = translate_page(page_base + page_size);
        int size_in_page = page_size - page_offset;
        util::safe_memcpy(translated_page + page_offset, word, size_in_page);
        util::safe_memcpy(next_translated_page, word + size_in_page, sizeof(T) - size_in_page);

        return;
    }

    util::safe_write<T>(translated_page + page_offset, value);
}

template<typename T>
void Mmu::blockify(reg_t address, size_t size, T&& lambda) {
    while (size != 0) {
        size_t block_size = get_block_size(address);
        if (block_size >= size) {
            lambda(address, size);
            return;
        }
        lambda(address, block_size);
        address += block_size;
        size -= block_size;
    }
}

void Mmu::copy_from_host(reg_t address, const void* target, size_t size) {
    const std::byte *target_bytes = reinterpret_cast<const std::byte*>(target);
    blockify(address, size, [=](reg_t start, size_t limit) {
        std::byte *buffer = translate_page(start &~ page_mask) + (start & page_mask);
        util::safe_memcpy(buffer, target_bytes + (start - address), limit);
    });
}

void Mmu::copy_to_host(reg_t address, void* target, size_t size) {
    std::byte *target_bytes = reinterpret_cast<std::byte*>(target);
    blockify(address, size, [=](reg_t start, size_t limit) {
        std::byte *buffer = translate_page(start &~ page_mask) + (start & page_mask);
        util::safe_memcpy(target_bytes + (start - address), buffer, limit);
    });
}

void Mmu::zero_memory(reg_t address, size_t size) {
    blockify(address, size, [=](reg_t start, size_t limit) {
        std::byte *buffer = translate_page(start &~ page_mask) + (start & page_mask);
        util::safe_memset(buffer, 0, limit);
    });
}

Paging_mmu::Paging_mmu() {

    // All valid entries stored in TLB will be page-aligned.
    // Since the bit pattern of -1 is page-aligned, the TLB is effectively marked as invalid.
    for (auto& query: cached_query) query = -1;
}

Paging_mmu::~Paging_mmu() {

    // Free any mapped pages.
    for (const auto& mapping: map) {
        munmap(mapping.second, page_size);
    }

    // Map the first page of emulated address space to the first page of virtual address space, so all null reference
    // can be catched.
    map[0] = 0;
}

void Paging_mmu::tlb_fetch(reg_t address) {

    // The input must be page-aligned.
    ASSERT((address & page_mask) == 0);

    const ptrdiff_t tag = (address >> log_page_size) & 31;

    // Unlikely event of cache miss. Get the entry in the map.
    const auto& mapping = map.find(address);

    // If the mapping does not exist, it will be a page fault.
    if (UNLIKELY(mapping == map.end())) {
        throw std::runtime_error {"Page fault"};
    }

    cached_query[tag] = address;
    cached_result[tag] = mapping->second;
}

void Paging_mmu::allocate_page(reg_t address, reg_t size) {

    // The input must be page-aligned.
    ASSERT((address & page_mask) == 0 && (size & page_mask) == 0);

    std::byte *result = reinterpret_cast<std::byte*>(
        mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0)
    );

    if (result == nullptr) {
        throw std::bad_alloc {};
    }

    for (reg_t offset = 0; offset < size; offset += page_size) {
        map[address + offset] = result + offset;
    }
}

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

    auto ptr = translate_page(address);

    void *ret = mmap(ptr, size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
    if (ret != ptr) {
        throw std::bad_alloc {};
    }

    return;
}

// As mentioned in mmu.h, we keep template interface for simplicity and tidyness, but the definition is in this file,
// so to make these function accessible we have to explicit instantiate them.
template uint16_t Mmu::load_memory_misaligned<uint16_t>(reg_t);
template uint32_t Mmu::load_memory_misaligned<uint32_t>(reg_t);
template uint64_t Mmu::load_memory_misaligned<uint64_t>(reg_t);
template void Mmu::store_memory_misaligned<uint16_t>(reg_t, uint16_t);
template void Mmu::store_memory_misaligned<uint32_t>(reg_t, uint32_t);
template void Mmu::store_memory_misaligned<uint64_t>(reg_t, uint64_t);

}
