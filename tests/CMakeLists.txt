add_custom_target(${ATL_TARGET_PREFIX}tests COMMENT "Build all the unit tests.")
add_dependencies(${ATL_TARGET_PREFIX}check ${ATL_TARGET_PREFIX}tests)

file(GLOB_RECURSE atl_unit_tests CONFIGURE_DEPENDS "*.cpp")
list(FILTER atl_unit_tests EXCLUDE REGEX "/persist/")
list(FILTER atl_unit_tests EXCLUDE REGEX "/(odr|consumer|portability)/")

foreach(_file IN LISTS atl_unit_tests)
    target_name_for(_target _output "${_file}")
    add_executable(${_target} "${_file}")
    set_target_properties(${_target} PROPERTIES OUTPUT_NAME ${_output})
    add_dependencies(${ATL_TARGET_PREFIX}tests ${_target})
    target_link_libraries(${_target} PUBLIC atl atl_experimental atl_catch2 atl_project_options)
    target_compile_definitions(${_target} PRIVATE ATL_ASSERT)
    set_target_properties(${_target} PROPERTIES CXX_VISIBILITY_PRESET hidden)
    add_test("tests/${_output}" ${_output})
endforeach()

add_dependencies(${ATL_TARGET_PREFIX}tests ${ATL_TARGET_PREFIX}headers-odr)
add_test(NAME tests/headers-odr COMMAND ${ATL_TARGET_PREFIX}headers-odr)

# PROBE (revert before merge): keep the output of a hang on MSVC x86
set_tests_properties(tests/nt-discrete_log PROPERTIES TIMEOUT 180)
