LD = g++-7
CXX = g++-7

LD_FLAGS = -g
CXX_FLAGS = -g -std=c++17 -fconcepts -Wall -Wextra -Iinclude/ -Og -fno-stack-protector

OBJS = \
	emu/elf_loader.o \
	emu/mmu.o \
	main.o \
	riscv/decoder.o \
	riscv/disassembler.o \
	softfp/float.o \
	util/format.o \
	util/safe_memory.o

default: all

.PHONY: all clean feature-test register unregister

all: codegen

clean:
	rm $(patsubst %,bin/%,$(OBJS) $(OBJS:.o=.d))

codegen: $(patsubst %,bin/%,$(OBJS)) $(LIBS)
	$(LD) $(LD_FLAGS) $^ -o $@

-include $(patsubst %,bin/%,$(OBJS:.o=.d))

# Special rule for feature testing
bin/feature.o: src/feature.cc
	@mkdir -p $(dir $@)
	$(CXX) -c -MMD -MP $(CXX_FLAGS) $< -o $@

# safe_memory.cc must be compiled with -fnon-call-exceptions for the program to work.
bin/util/safe_memory.o: src/util/safe_memory.cc
	@mkdir -p $(dir $@)
	$(CXX) -c -MMD -MP $(CXX_FLAGS) -fnon-call-exceptions $< -o $@

bin/%.o: src/%.cc bin/feature.o
	@mkdir -p $(dir $@)
	$(CXX) -c -MMD -MP $(CXX_FLAGS) $< -o $@

register: codegen
	sudo bash -c "echo ':riscv:M::\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\xf3\x00:\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff\xff:$(shell realpath codegen):' > /proc/sys/fs/binfmt_misc/register"

unregister:
	sudo bash -c "echo -1 > /proc/sys/fs/binfmt_misc/riscv"
