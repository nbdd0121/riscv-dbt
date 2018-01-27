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

public:
    virtual std::byte* translate(reg_t address) = 0;

    virtual void allocate_page(reg_t address, reg_t size) = 0;

    template<typename T>
    T load_memory(reg_t address) {
        return util::safe_read<T>(translate(address));
    }

    template<typename T>
    void store_memory(reg_t address, T value) {
        util::safe_write<T>(translate(address), value);
    }

    void copy_from_host(reg_t address, const void* target, size_t size) {
        util::safe_memcpy(translate(address), target, size);
    }

    void copy_to_host(reg_t address, void* target, size_t size) {
        util::safe_memcpy(target, translate(address), size);
    }

    void zero_memory(reg_t address, size_t size) {
        util::safe_memset(translate(address), 0, size);
    }
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
    virtual std::byte* translate(reg_t address) override {
        return memory_ + address;
    }

    // Allocate memory at given virtual address and estabilish mapping.
    // The address and size must be page-aligned.
    virtual void allocate_page(reg_t address, reg_t size) override;
};

// This MMU performs identity mapping. The address space of the program and the emulator must not overlap.
class Id_mmu final: public Mmu {
public:
    virtual std::byte* translate(reg_t address) override {
        return reinterpret_cast<std::byte*>(address);
    }

    // Allocate memory at given virtual address and estabilish mapping.
    // The address and size must be page-aligned.
    virtual void allocate_page(reg_t address, reg_t size) override;
};

}

#endif
