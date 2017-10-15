#ifndef UTIL_BITFIELD_H
#define UTIL_BITFIELD_H

#include <type_traits>

namespace util {

// It a quite common operation to view an integer as bits and extract from or pack fields into the integer. Doing these
// operation by shift and bit operation each time is tedious, inelegant, and prone to errors. Bitfield class provide a
// easy way to access them.
// E.g. Bitfield<int, 23, 16, 7, 0> will extract 0x3478 from 0x12345678.
template<typename Type, int... Range>
class Bitfield {

    // Bitfield should be only operating on unsigned types.
    static_assert(std::is_unsigned<Type>::value, "Bitfield must be used on unsigned types.");

    // This unspecialized class catches the case that sizeof...(Range) is 0 or odd, but odd number does not make sense.
    static_assert(sizeof...(Range) == 0, "Bitfield must have an even number of integer arguments.");

public:
    static constexpr int width = 0;
    static constexpr Type extract(Type) noexcept { return 0; }
    static constexpr Type pack(Type bits, Type) noexcept { return bits; }
};

template<typename Type, int Hi, int Lo, int... Range>
class Bitfield<Type, Hi, Lo, Range...> {
    static_assert(Hi >= Lo, "Hi must be >= Lo.");

    // This class will handle only one segment, and then it passes the task to a smaller bitfield.
    using Chain = Bitfield<Type, Range...>;

    // Mask containing ones on all bits within range [Lo, Hi]
    static constexpr Type mask = ((static_cast<Type>(1) << (Hi - Lo + 1)) - 1) << Lo;

public:
    static constexpr int width = Chain::width + (Hi - Lo + 1);

    static constexpr Type extract(Type bits) noexcept {
        return Chain::extract(bits) | ((bits & mask) >> Lo << Chain::width);
    }

    static constexpr Type pack(Type bits, Type value) noexcept {
        return Chain::pack((bits & ~mask) | ((value >> Chain::width << Lo) & mask), value);
    }
};

}

#endif
