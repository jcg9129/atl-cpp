# The parent project never acquires documentation targets or dependencies.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(atl_docs_dir "${PROJECT_BINARY_DIR}/docs")
set(atl_coverage_page "${atl_docs_dir}/coverage.md")
set(atl_coverage_command "${Python3_EXECUTABLE}"
    "${PROJECT_SOURCE_DIR}/scripts/check-docs.py"
    --root "${PROJECT_SOURCE_DIR}" --output "${atl_coverage_page}")

if(BUILD_TESTS)
    add_test(NAME docs/checker COMMAND "${Python3_EXECUTABLE}" "${PROJECT_SOURCE_DIR}/scripts/test-check-docs.py")
    add_test(NAME docs/coverage COMMAND ${atl_coverage_command})
    file(GLOB_RECURSE atl_doc_examples CONFIGURE_DEPENDS
        "${PROJECT_SOURCE_DIR}/docs/examples/*.cpp")
    foreach(source IN LISTS atl_doc_examples)
        file(RELATIVE_PATH example "${PROJECT_SOURCE_DIR}/docs/examples" "${source}")
        string(REGEX REPLACE "\\.cpp$" "" example "${example}")
        string(REPLACE "/" "-" target "atl_doc_${example}")
        set(expected "${PROJECT_SOURCE_DIR}/docs/examples/${example}.out")
        if(NOT EXISTS "${expected}")
            message(FATAL_ERROR "Documentation example lacks expected stdout: ${expected}")
        endif()
        add_executable(${target} "${source}")
        target_link_libraries(${target} PRIVATE atl atl_project_options)
        target_compile_definitions(${target} PRIVATE ATL_ASSERT)
        add_dependencies(${ATL_TARGET_PREFIX}tests ${target})
        add_test(NAME "docs/example/${example}" COMMAND "${CMAKE_COMMAND}"
            "-DBUILD_DIR=${PROJECT_BINARY_DIR}" "-DTARGET=${target}"
            "-DCONFIG=$<CONFIG>" "-DEXECUTABLE=$<TARGET_FILE:${target}>"
            "-DEXPECTED=${expected}" "-DACTUAL=${PROJECT_BINARY_DIR}/${target}.out"
            -P "${PROJECT_SOURCE_DIR}/cmake/RunDocExample.cmake")
        # CTest may run in parallel, but two invocations must not build one tree.
        set_tests_properties("docs/example/${example}" PROPERTIES
            RESOURCE_LOCK atl_doc_example_build TIMEOUT 120)
    endforeach()
endif()

if(NOT ATL_BUILD_DOCS)
    return()
endif()
find_package(Doxygen 1.15 REQUIRED)
set(ATL_DOXYGEN_AWESOME_DIR "" CACHE PATH "Extracted doxygen-awesome-css checkout (empty for plain HTML)")
file(MAKE_DIRECTORY "${atl_docs_dir}")
file(GLOB atl_guide_pages CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/docs/*.md")
set(ATL_DOXYGEN_INPUT "\"${PROJECT_SOURCE_DIR}/include/atl\" \"${PROJECT_SOURCE_DIR}/README.md\"")
foreach(page IN LISTS atl_guide_pages)
    string(APPEND ATL_DOXYGEN_INPUT " \"${page}\"")
endforeach()
string(APPEND ATL_DOXYGEN_INPUT " \"${atl_coverage_page}\"")
file(WRITE "${atl_docs_dir}/namespaces.dox"
    "/** @namespace atl\n * @brief Public algorithms, types and utilities.\n */\n"
    "/** @namespace atl::graph\n * @brief Graph vocabulary, storage, traversal and graph algorithms.\n */\n")
string(APPEND ATL_DOXYGEN_INPUT " \"${atl_docs_dir}/namespaces.dox\"")
set(ATL_DOXYGEN_STYLES "")
set(ATL_DOXYGEN_FILES "\"${PROJECT_SOURCE_DIR}/LICENSE\"")
set(ATL_DOXYGEN_HEADER "")
if(ATL_DOXYGEN_AWESOME_DIR)
    get_filename_component(ATL_DOXYGEN_AWESOME_DIR "${ATL_DOXYGEN_AWESOME_DIR}" ABSOLUTE)
    foreach(asset IN ITEMS doxygen-awesome.css doxygen-awesome-sidebar-only.css
            doxygen-awesome-sidebar-only-darkmode-toggle.css doxygen-awesome-darkmode-toggle.js)
        if(NOT EXISTS "${ATL_DOXYGEN_AWESOME_DIR}/${asset}")
            message(FATAL_ERROR "Theme asset missing: ${ATL_DOXYGEN_AWESOME_DIR}/${asset}")
        endif()
        if(asset MATCHES "\\.css$")
            string(APPEND ATL_DOXYGEN_STYLES " \"${ATL_DOXYGEN_AWESOME_DIR}/${asset}\"")
        else()
            string(APPEND ATL_DOXYGEN_FILES " \"${ATL_DOXYGEN_AWESOME_DIR}/${asset}\"")
        endif()
    endforeach()
    # Distribute the theme notice with the generated assets, using a distinct
    # filename so README's LICENSE link still refers to atl's 0BSD license.
    if(NOT EXISTS "${ATL_DOXYGEN_AWESOME_DIR}/LICENSE")
        message(FATAL_ERROR "Theme license missing: ${ATL_DOXYGEN_AWESOME_DIR}/LICENSE")
    endif()
    configure_file("${ATL_DOXYGEN_AWESOME_DIR}/LICENSE"
        "${atl_docs_dir}/doxygen-awesome-LICENSE" COPYONLY)
    string(APPEND ATL_DOXYGEN_FILES " \"${atl_docs_dir}/doxygen-awesome-LICENSE\"")
    # Generate Doxygen's own version-matched header in the build tree, then add
    # the toggle hook and a title fix (the main page is titled by README's "# atl",
    # which would read "atl: atl"). No third-party templates or assets enter the source tree.
    execute_process(COMMAND "${DOXYGEN_EXECUTABLE}" -w html
        "${atl_docs_dir}/header.html" "${atl_docs_dir}/footer.html" "${atl_docs_dir}/default.css"
        COMMAND_ERROR_IS_FATAL ANY OUTPUT_QUIET)
    file(READ "${atl_docs_dir}/header.html" header)
    string(REPLACE "</head>"
        "<script src=\"$relpath^doxygen-awesome-darkmode-toggle.js\"></script>\n<script>DoxygenAwesomeDarkModeToggle.init()</script>\n<script>if (document.title === \"$projectname: $projectname\") document.title = \"$projectname\";</script>\n</head>" header "${header}")
    file(WRITE "${atl_docs_dir}/header.html" "${header}")
    set(ATL_DOXYGEN_HEADER "${atl_docs_dir}/header.html")
endif()
configure_file("${PROJECT_SOURCE_DIR}/docs/Doxyfile.in" "${atl_docs_dir}/Doxyfile" @ONLY)
add_custom_target(atl_docs
    COMMAND "${CMAKE_COMMAND}" "-DPYTHON=${Python3_EXECUTABLE}"
        "-DSOURCE_DIR=${PROJECT_SOURCE_DIR}" "-DDOCS_DIR=${atl_docs_dir}"
        "-DDOXYGEN=${DOXYGEN_EXECUTABLE}"
        -P "${PROJECT_SOURCE_DIR}/cmake/BuildDocs.cmake"
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    COMMENT "Checking guide coverage and generating the public API site"
    VERBATIM)
