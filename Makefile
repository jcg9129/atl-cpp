# Shared CMake presets are the source of truth for local builds and CI.
SHELL := /bin/bash
PRESET ?= gcc
JOBS ?= $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 2)
CLANG_FORMAT ?= clang-format-22
CLANG_TIDY ?= clang-tidy-22
export PRESET JOBS CLANG_FORMAT CLANG_TIDY

.DEFAULT_GOAL := help
.PHONY: help configure test sanitize fmt fmt-fix tidy tidy-proof headers install-check check clean-build docs

help:
	@echo 'make test PRESET=gcc|clang|appleclang|fallbacks; make sanitize'
	@echo 'make docs [ATL_DOXYGEN_AWESOME_DIR=/path/to/theme]'
	@echo 'make fmt | fmt-fix | tidy | headers | install-check | check | clean-build'
	@echo 'Override PRESET, JOBS, CLANG_FORMAT, or CLANG_TIDY as needed.'

configure:
	cmake --preset $(PRESET)

test: configure
	cmake --build --preset $(PRESET) --parallel $(JOBS)
	ctest --preset $(PRESET)

docs:
	cmake --preset docs -DATL_DOXYGEN_AWESOME_DIR="$(ATL_DOXYGEN_AWESOME_DIR)"
	cmake --build --preset docs --parallel 1

sanitize:
	UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 $(MAKE) test PRESET=sanitizers

fmt fmt-fix tidy tidy-proof:
	@bash scripts/style.sh $@

headers: configure
	cmake --build build/$(PRESET) --target headers headers-odr --parallel $(JOBS)

install-check: configure
	@bash scripts/install-check.sh $(PRESET)

check: fmt tidy
	$(MAKE) test headers PRESET=gcc

# Accept only a single safe preset name, never an arbitrary path or symlink.
clean-build:
	@case "$$PRESET" in ''|*[!a-zA-Z0-9_-]*) echo 'Unsafe PRESET' >&2; exit 1;; esac; \
	if [ -L build ] || [ -L "build/$$PRESET" ]; then echo 'Refusing symlink' >&2; exit 1; fi; \
	path="build/$$PRESET"; \
	case "$$path" in build/?*) rm -rf -- "$$path";; *) echo 'Unsafe build path' >&2; exit 1;; esac
