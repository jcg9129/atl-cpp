# Produce the site and diagnostics even when coverage has gaps, then fail CI.
execute_process(COMMAND "${PYTHON}" "${SOURCE_DIR}/scripts/check-docs.py"
    --root "${SOURCE_DIR}" --output "${DOCS_DIR}/coverage.md"
    RESULT_VARIABLE coverage_result)
file(REMOVE_RECURSE "${DOCS_DIR}/html")
execute_process(COMMAND "${DOXYGEN}" "${DOCS_DIR}/Doxyfile"
    WORKING_DIRECTORY "${SOURCE_DIR}" RESULT_VARIABLE doxygen_result)
if(NOT coverage_result EQUAL 0 OR NOT doxygen_result EQUAL 0)
    if(EXISTS "${DOCS_DIR}/warnings.log")
        file(READ "${DOCS_DIR}/warnings.log" warnings)
        message("${warnings}")
    endif()
    message(FATAL_ERROR "Documentation failed: coverage=${coverage_result}, Doxygen=${doxygen_result}. See ${DOCS_DIR}/warnings.log")
endif()
