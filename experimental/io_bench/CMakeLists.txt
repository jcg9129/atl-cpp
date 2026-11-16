include(CheckCXXSourceCompiles)

# Compile and link probes avoid registering or running benchmark workloads.
check_cxx_source_compiles([=[
    #include <print>
    int main() { std::println("{}", 1); }
]=] ATL_IO_HARNESS_HAS_PRINTLN)
check_cxx_source_compiles([=[
    #include <cstdio>
    int main() {
    #ifdef _WIN32
        return _putchar_nolock(_getchar_nolock());
    #else
        return putchar_unlocked(getchar_unlocked());
    #endif
    }
]=] ATL_IO_HARNESS_HAS_UNLOCKED_STDIO)

add_custom_target(io-harness COMMENT "Build the standalone I/O benchmark harness.")

function(atl_add_io_harness name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} PRIVATE atl atl_project_options)
    target_compile_definitions(${name} PRIVATE ${ARGN})
    if(ATL_BENCHMARK_ASSERTS)
        target_compile_definitions(${name} PRIVATE ATL_ASSERT)
    endif()
    add_dependencies(io-harness ${name})
endfunction()

atl_add_io_harness(io-bench-generator gen.cpp)

set(atl_io_backends cin atl)
if(ATL_IO_HARNESS_HAS_PRINTLN)
    list(APPEND atl_io_backends println)
else()
    message(STATUS "I/O harness: skipping println backends (std::println unavailable)")
endif()
if(ATL_IO_HARNESS_HAS_UNLOCKED_STDIO)
    list(APPEND atl_io_backends getchar)
else()
    message(STATUS "I/O harness: skipping getchar backends (unlocked stdio unavailable)")
endif()

foreach(backend IN LISTS atl_io_backends)
    string(TOUPPER "${backend}" backend_macro)
    atl_add_io_harness(io-bench-range-${backend} bench.cpp IO_${backend_macro})
    atl_add_io_harness(io-bench-read-${backend} micro.cpp IO_${backend_macro} BENCH_READ)
    atl_add_io_harness(io-bench-write-${backend} micro.cpp IO_${backend_macro} BENCH_WRITE)
endforeach()

# These configurations match the extra variants in run.py and run_micro.py.
atl_add_io_harness(io-bench-range-atl-mmap bench.cpp IO_ATL ATL_USE_MMAP)
atl_add_io_harness(io-bench-write-atl-lut4 micro.cpp IO_ATL ATL_IO_BIG_LUT BENCH_WRITE)
