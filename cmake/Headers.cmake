# Each public header gets its own TU: an aggregate include can hide missing includes.
file(GLOB_RECURSE atl_public_headers CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/include/atl/*.hpp")
list(SORT atl_public_headers)
set(ATL_ALL_HEADERS_INCLUDES "")
set(header_sources "")
foreach(header IN LISTS atl_public_headers)
    file(RELATIVE_PATH relative "${PROJECT_SOURCE_DIR}/include" "${header}")
    string(APPEND ATL_ALL_HEADERS_INCLUDES "#include <${relative}>\n")
    set(source "${CMAKE_CURRENT_BINARY_DIR}/headers/${relative}.cpp")
    set(content "#include <${relative}>\n")
    # Outside the software 128-bit headers themselves, no header may compile the
    # software classes where native 128-bit integers are the aliases.
    if(NOT relative MATCHES "^atl/types/(int128|detail/software_int128_format)\\.hpp$")
        string(APPEND content "#include <atl/config.hpp>\n#if ATL_HAS_INT128 && defined(ATL_DETAIL_SOFTWARE_INT128)\n"
               "#error \"${relative} includes the software 128-bit classes on a native target\"\n#endif\n")
    endif()
    file(WRITE "${source}" "${content}")
    list(APPEND header_sources "${source}")
endforeach()
foreach(mode IN ITEMS normal no-int128)
    add_library(atl_headers_${mode} OBJECT EXCLUDE_FROM_ALL ${header_sources})
    target_link_libraries(atl_headers_${mode} PRIVATE atl atl_project_options)
    target_compile_definitions(atl_headers_${mode} PRIVATE ATL_ASSERT)
endforeach()
target_compile_definitions(atl_headers_no-int128 PRIVATE ATL_NO_INT128)
add_custom_target(${ATL_TARGET_PREFIX}headers-normal DEPENDS atl_headers_normal)
add_custom_target(${ATL_TARGET_PREFIX}headers-no-int128 DEPENDS atl_headers_no-int128)
add_custom_target(${ATL_TARGET_PREFIX}headers DEPENDS ${ATL_TARGET_PREFIX}headers-normal ${ATL_TARGET_PREFIX}headers-no-int128)

configure_file("${PROJECT_SOURCE_DIR}/tests/odr/all_headers.hpp.in"
               "${CMAKE_CURRENT_BINARY_DIR}/odr/all_headers.hpp" @ONLY)
add_executable(${ATL_TARGET_PREFIX}headers-odr EXCLUDE_FROM_ALL
    "${PROJECT_SOURCE_DIR}/tests/odr/tu1.cpp" "${PROJECT_SOURCE_DIR}/tests/odr/tu2.cpp"
    "${PROJECT_SOURCE_DIR}/tests/odr/split_narrow.cpp" "${PROJECT_SOURCE_DIR}/tests/odr/split_full.cpp")
target_include_directories(${ATL_TARGET_PREFIX}headers-odr PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/odr")
target_link_libraries(${ATL_TARGET_PREFIX}headers-odr PRIVATE atl atl_project_options)
target_compile_definitions(${ATL_TARGET_PREFIX}headers-odr PRIVATE ATL_ASSERT)
if(BUILD_TESTS)
    set_target_properties(${ATL_TARGET_PREFIX}headers-odr PROPERTIES EXCLUDE_FROM_ALL FALSE)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)
# Git is only needed for this development regression check, never by consumers.
add_custom_target(${ATL_TARGET_PREFIX}headers-compat
    COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/scripts/check-header-compat.py"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}" VERBATIM)
add_dependencies(${ATL_TARGET_PREFIX}headers ${ATL_TARGET_PREFIX}headers-compat)
