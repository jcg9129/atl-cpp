cmake_minimum_required(VERSION 3.25)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

project(atl VERSION 0.1.0 LANGUAGES CXX)

# Default to optimization and debug info for standalone, single-config builds.
if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE
        RelWithDebInfo
        CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release
                                                 RelWithDebInfo MinSizeRel)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS on)
set(CMAKE_CXX_EXTENSIONS off)

# Default standard for any target that isn't wired to `atl` (e.g. vendored
# Catch2). Targets linking `atl` inherit C++23 via target_compile_features below.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED on)

include(Utils)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_program(ATL_CCACHE_PROGRAM ccache)
set(atl_ccache_default OFF)
if(PROJECT_IS_TOP_LEVEL AND ATL_CCACHE_PROGRAM)
    set(atl_ccache_default ON)
endif()
option(ENABLE_CCACHE "Use ccache when available" ${atl_ccache_default})
if(ENABLE_CCACHE AND NOT DEFINED CMAKE_CXX_COMPILER_LAUNCHER
   AND NOT DEFINED CACHE{CMAKE_CXX_COMPILER_LAUNCHER})
    if(NOT ATL_CCACHE_PROGRAM)
        message(FATAL_ERROR "ENABLE_CCACHE requires ccache on PATH")
    endif()
    set(CMAKE_CXX_COMPILER_LAUNCHER "${ATL_CCACHE_PROGRAM}")
endif()

# Options

option(ENABLE_ASAN "compile with address sanitizer enabled")
option(ENABLE_MSAN "compile with memory sanitizer enabled")
option(ENABLE_LSAN "compile with leak sanitizer enabled")
option(ENABLE_UBSAN "compile with undefined behavior sanitizer enabled")
option(ENABLE_COVERAGE "compile with test coverage support")
option(DISABLE_WERROR "Do not treat project warnings as errors")

option(ENABLE_BOOST_COROUTINE "run benchmarks with boost coroutine" off)

option(ENABLE_PORTABLE_FALLBACKS
       "force the C++ feature fallbacks (ATL_FORCE_FALLBACKS) to test portable paths"
       off)

option(BUILD_TESTS "Build tests" ${PROJECT_IS_TOP_LEVEL})
option(BUILD_BENCHMARKS "Build benchmarks" ${PROJECT_IS_TOP_LEVEL})
option(ATL_INSTALL "Install atl headers and package" ${PROJECT_IS_TOP_LEVEL})
if(PROJECT_IS_TOP_LEVEL)
    option(ATL_BUILD_DOCS "Build the documentation site" OFF)
endif()

# Keep explicitly enabled development targets out of a parent project's namespace.
set(ATL_TARGET_PREFIX "atl_")
if(PROJECT_IS_TOP_LEVEL)
    set(ATL_TARGET_PREFIX "")
endif()

option(ENABLE_NO_INT128 "Force ATL_NO_INT128" OFF)
set(ATL_EXPECT_ARCH "" CACHE STRING "Expected architecture: x64, x86, or arm64")

# Tests enable atl_assert; benchmarks disable it by default to measure release behavior.
option(ATL_BENCHMARK_ASSERTS "build benchmarks with atl_assert enabled" OFF)

# Project-wide development flags apply to tests and benchmarks via linkage;
# downstream consumers of `atl` do not inherit -Werror or sanitizers.
add_library(atl_project_options INTERFACE)

if(MSVC)
    target_compile_options(atl_project_options INTERFACE /W4 /permissive-
                           /Zc:__cplusplus /Zc:preprocessor /utf-8)
    if(NOT DISABLE_WERROR)
        target_compile_options(atl_project_options INTERFACE /WX)
    endif()
else()
    target_compile_options(
        atl_project_options
        INTERFACE -Wall
                  -Wextra
                  -pedantic
                  -Wno-unused-parameter
                  -Wno-extended-offsetof
                  -Wno-c++23-extensions
                  -Wno-c2y-extensions
                  -Wno-unknown-warning-option
                  -Wno-type-limits)
    if(NOT DISABLE_WERROR)
        target_compile_options(atl_project_options INTERFACE -Werror)
    endif()
endif()

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    target_compile_options(atl_project_options INTERFACE -Qunused-arguments)
endif()

# Sanitizers / coverage: instrument both compile and link steps.
if(ENABLE_ASAN)
    if(MSVC)
        target_compile_options(atl_project_options INTERFACE /fsanitize=address)
    else()
        target_compile_options(atl_project_options INTERFACE -fsanitize=address)
        target_link_options(atl_project_options INTERFACE -fsanitize=address)
    endif()
endif()
if(ENABLE_LSAN)
    target_compile_options(atl_project_options INTERFACE -fsanitize=leak)
    target_link_options(atl_project_options INTERFACE -fsanitize=leak)
endif()
if(ENABLE_UBSAN)
    target_compile_options(atl_project_options INTERFACE -fsanitize=undefined)
    target_link_options(atl_project_options INTERFACE -fsanitize=undefined)
endif()
if(ENABLE_MSAN)
    target_compile_options(atl_project_options INTERFACE -fsanitize=memory)
    target_link_options(atl_project_options INTERFACE -fsanitize=memory)
endif()
if(ENABLE_COVERAGE)
    target_compile_options(atl_project_options INTERFACE --coverage)
    target_link_options(atl_project_options INTERFACE --coverage)
endif()

if(ENABLE_NO_INT128)
    target_compile_definitions(atl_project_options INTERFACE ATL_NO_INT128)
endif()

# Force the portable (feature-fallback) code paths, to exercise them on a
# toolchain that actually has the features.
if(ENABLE_PORTABLE_FALLBACKS)
    target_compile_definitions(atl_project_options INTERFACE ATL_FORCE_FALLBACKS)
endif()

# library

add_library(atl INTERFACE)
add_library(atl::atl ALIAS atl)

target_compile_features(atl INTERFACE cxx_std_23)

target_include_directories(
    atl
    INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
              $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Experimental / prototype implementations, kept out of the public `atl` library
# (namespace atl::experimental). Available to benchmarks and tests for A/B work.
add_library(atl_experimental INTERFACE)
target_link_libraries(atl_experimental INTERFACE atl)
target_include_directories(atl_experimental
                           INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/experimental>)

# Catch2 is shared by all tests and benchmarks, including benchmark-only builds.
if(BUILD_TESTS OR BUILD_BENCHMARKS)
    add_library(atl_catch2 STATIC "${PROJECT_SOURCE_DIR}/external/catch2/catch.cpp")
    target_include_directories(atl_catch2 SYSTEM PUBLIC "${PROJECT_SOURCE_DIR}/external")
    # Vendored Catch2's Optional<T> is padded under x86 alignment (C4324, informational); the
    # warning fires in our test translation units, which /WX would turn into an error.
    target_compile_options(atl_catch2 PUBLIC $<$<CXX_COMPILER_ID:MSVC>:/wd4324>)
endif()

if(PROJECT_IS_TOP_LEVEL OR BUILD_TESTS OR BUILD_BENCHMARKS)
    include(Headers)

    if(BUILD_TESTS)
        add_custom_target(
            ${ATL_TARGET_PREFIX}check
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure --no-tests=error
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            COMMENT "Build and run all the tests.")
    endif()

    if(BUILD_TESTS)
        enable_testing()
        add_subdirectory(tests)
    endif()

    if(BUILD_BENCHMARKS)
        add_subdirectory(benchmarks)
    endif()

    # A portable smoke test independent of Catch2 and the full test suite.
    add_executable(${ATL_TARGET_PREFIX}architecture-check EXCLUDE_FROM_ALL tests/portability/architecture.cpp)
    target_link_libraries(${ATL_TARGET_PREFIX}architecture-check PRIVATE atl atl_project_options)
    if(ATL_EXPECT_ARCH)
        if(NOT ATL_EXPECT_ARCH MATCHES "^(x64|x86|arm64)$")
            message(FATAL_ERROR "Unknown ATL_EXPECT_ARCH: ${ATL_EXPECT_ARCH}")
        endif()
        target_compile_definitions(${ATL_TARGET_PREFIX}architecture-check PRIVATE "ATL_EXPECT_${ATL_EXPECT_ARCH}=1")
    endif()
    add_executable(${ATL_TARGET_PREFIX}no-int128-check EXCLUDE_FROM_ALL tests/portability/no_int128.cpp)
    target_link_libraries(${ATL_TARGET_PREFIX}no-int128-check PRIVATE atl atl_project_options)
    target_compile_definitions(${ATL_TARGET_PREFIX}no-int128-check PRIVATE ATL_NO_INT128 ATL_ASSERT)
    add_custom_target(${ATL_TARGET_PREFIX}portability DEPENDS ${ATL_TARGET_PREFIX}headers ${ATL_TARGET_PREFIX}headers-odr ${ATL_TARGET_PREFIX}architecture-check ${ATL_TARGET_PREFIX}no-int128-check)
    if(BUILD_TESTS)
        add_dependencies(${ATL_TARGET_PREFIX}portability ${ATL_TARGET_PREFIX}tests)
    endif()
endif()

if(PROJECT_IS_TOP_LEVEL AND (BUILD_TESTS OR ATL_BUILD_DOCS))
    include(Documentation)
endif()

if(ATL_INSTALL)
    install(TARGETS atl EXPORT atlTargets)
    install(DIRECTORY include/atl DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" FILES_MATCHING PATTERN "*.hpp")
    install(EXPORT atlTargets NAMESPACE atl:: DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/atl")
    configure_package_config_file(cmake/atlConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/atlConfig.cmake"
        INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/atl")
    write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/atlConfigVersion.cmake"
        VERSION "${PROJECT_VERSION}" COMPATIBILITY SameMajorVersion ARCH_INDEPENDENT)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/atlConfig.cmake"
                  "${CMAKE_CURRENT_BINARY_DIR}/atlConfigVersion.cmake"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/atl")
endif()
