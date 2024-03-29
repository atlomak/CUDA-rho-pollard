# Compiler
NVCC := nvcc
# Flags
NVCCFLAGS := -std=c++11

# Directories
SRCDIR := src
BUILDDIR := build

# Find all CUDA source files
CUFILES := $(wildcard $(SRCDIR)/*.cu)

# Generate corresponding executable names
EXECUTABLES := $(patsubst $(SRCDIR)/%.cu,$(BUILDDIR)/%,$(CUFILES))

# Rule for compiling CUDA files
$(BUILDDIR)/%: $(SRCDIR)/%.cu
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

# Rule for building all executables
all: $(EXECUTABLES)

# Rule for cleaning
clean:
	@rm -rf $(BUILDDIR)

.PHONY: all clean
