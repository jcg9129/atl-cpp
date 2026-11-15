add_custom_target(${ATL_TARGET_PREFIX}benchmarks COMMENT "Build all benchmarks.")

file(GLOB_RECURSE atl_benchmarks CONFIGURE_DEPENDS "*.cpp")
foreach(_file IN LISTS atl_benchmarks)
    target_name_for(_target _output "${_file}")
    add_executable(${_target} "${_file}")
    add_dependencies(${ATL_TARGET_PREFIX}benchmarks ${_target})
    set_target_properties(${_target} PROPERTIES OUTPUT_NAME ${_output})
    target_link_libraries(${_target} PUBLIC atl atl_experimental atl_catch2 atl_project_options)
    if(ATL_BENCHMARK_ASSERTS)
        target_compile_definitions(${_target} PRIVATE ATL_ASSERT)
    endif()
endforeach()

# Create a custom target that will run all benchmarks
add_custom_target(${ATL_TARGET_PREFIX}run_benchmarks
    COMMAND ${CMAKE_COMMAND} -E echo "Running all benchmarks..."
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${ATL_TARGET_PREFIX}benchmarks
    COMMENT "Running all benchmarks"
)

# Add commands to run each benchmark
foreach(_file IN LISTS atl_benchmarks)
    target_name_for(_target _output "${_file}")
    add_custom_target(${ATL_TARGET_PREFIX}run_${_target}
        COMMAND ${CMAKE_COMMAND} -E echo "Running ${_output}..."
        COMMAND ./${_output}
        DEPENDS ${_target}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    add_dependencies(${ATL_TARGET_PREFIX}run_benchmarks ${ATL_TARGET_PREFIX}run_${_target})
endforeach()
