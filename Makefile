
SHADERC ?= glslc
SDIR := res/shaders
SHADERFLAGS := -I $(SDIR) --target-env=vulkan1.1

SHADERSRCS := $(wildcard $(SDIR)/*.frag) $(wildcard $(SDIR)/*.vert)
SHADERBINS := $(addsuffix .spv, $(SHADERSRCS))

.PHONY: all shaders debug release debug-run release-run clippy

all: shaders

shaders: $(SHADERBINS)

debug: shaders
    cargo build

release: shaders
    cargo build --release

debug-run: shaders
    cargo run

release-run: shaders
    cargo run --release

clippy:
    cargo clippy --release

%.frag.spv : %.frag
	@$(SHADERC) $(SHADERFLAGS) $< -o $@
	@echo glslc $<

%.vert.spv : %.vert
	@$(SHADERC) $(SHADERFLAGS) $< -o $@
	@echo glslc $<

