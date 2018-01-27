#ifndef EMU_MMU_H
#define EMU_MMU_H

#include <cstddef>
#include <stdexcept>
#include <tuple>
#include <unordered_map>

#include "config.h"
#include "emu/typedef.h"
#include "util/assert.h"
#include "util/safe_memory.h"

namespace emu {

static constexpr reg_t page_size = 0x1000;
static constexpr reg_t page_mask = page_size - 1;
static constexpr reg_t log_page_size = 12;

class Mmu {
public:
    virtual ~Mmu() {}

private:
    // Used internally to implement copy_*_host and zero_memory.
    template<typename T>
    void blockify(reg_t address, size_t size, T&& lambda);

public:
    virtual std::byte* translate_page(reg_t address) = 0;
    
    // Get the size of continuous block starting from the given address.
    virtual size_t get_block_size(reg_t address) = 0;

    virtual void allocate_page(reg_t address, reg_t size) = 0;

    template<typename T>
    T load_memory_aligned(reg_t address) {
        reg_t page_offset = address & page_mask;
        reg_t page_base = address &~ page_mask;
        std::byte *translated_page = translate_page(page_base);
        return util::safe_read<T>(translated_page + page_offset);
    }

    template<typename T>
    void store_memory_aligned(reg_t address, T value) {
        reg_t page_offset = address & page_mask;
        reg_t page_base = address &~ page_mask;
        std::byte *translated_page = translate_page(page_base);
        util::safe_write<T>(translated_page + page_offset, value);
    }

    // To make the interface obvious and tidy, these functions are defined as templates. However the function lies on a
    // very cold path, and is relatively complex. Putting it inside header file might cause compiler to inline the file
    // and/or make the compilation time longer.
    template<typename T>
    T load_memory_misaligned(reg_t address);

    template<typename T>
    void store_memory_misaligned(reg_t address, T value);

    template<typename T>
    T load_memory(reg_t address) {
        constexpr reg_t addr_mask = sizeof(T) - 1;
        if (LIKELY((address & addr_mask) == 0)) {
            return load_memory_aligned<T>(address);
        } else {
            return load_memory_misaligned<T>(address);
        }
    }

    template<typename T>
    void store_memory(reg_t address, T value) {
        constexpr reg_t addr_mask = sizeof(T) - 1;
        if (LIKELY((address & addr_mask) == 0)) {
            return store_memory_aligned<T>(address, value);
        } else {
            return store_memory_misaligned<T>(address, value);
        }
    }

    void copy_from_host(reg_t address, const void* source, size_t size);
    void copy_to_host(reg_t address, void* target, size_t size);
    void zero_memory(reg_t address, size_t size);
};

class Flat_mmu final: public Mmu {
public:

    // These fields are exposed as public intentionally for now, since they are needed by the code generator.
    std::byte *memory_;
    size_t size_;

    // Similarly, placed in mmu.cc since it contains mmap/munmap.
    Flat_mmu(size_t size);
    ~Flat_mmu();

public:
    // Translating address of a page. The address must be page-aligned.
    virtual std::byte* translate_page(reg_t address) override {

        // The input must be page-aligned.
        ASSERT((address & page_mask) == 0);

        if (UNLIKELY(address >= size_)) {
            throw std::runtime_error {"Page fault"};
        }

        return memory_ + address;
    }

    // In Flat_mmu's case we divide the memory space into two regions: valid and invalid.
    virtual size_t get_block_size(reg_t address) override {
        if (address < size_) {
            return size_ - address;
        } else {
            return page_size - (address &~ page_mask);
        }
    }

    // Allocate memory at given virtual address and estabilish mapping.
    // The address and size must be page-aligned.
    virtual void allocate_page(reg_t address, reg_t size) override;
};

// This MMU performs identity mapping. The address space of the program and the emulator must not overlap.
class Id_mmu final: public Mmu {
public:
    // Translating address of a page. The address must be page-aligned.
    virtual std::byte* translate_page(reg_t address) override {

        // The input must be page-aligned.
        ASSERT((address & page_mask) == 0);

        return reinterpret_cast<std::byte*>(address);
    }

    // In Id_mmu's case we divide the memory space into two regions: valid and invalid.
    virtual size_t get_block_size(reg_t address) override {
        return 0x800000000000 - address;
    }

    // Allocate memory at given virtual address and estabilish mapping.
    // The address and size must be page-aligned.
    virtual void allocate_page(reg_t address, reg_t size) override;
};

}

#endif
