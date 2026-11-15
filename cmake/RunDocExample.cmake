execute_process(COMMAND "${CMAKE_COMMAND}" --build "${BUILD_DIR}"
    --target "${TARGET}" --config "${CONFIG}" --parallel 1
    RESULT_VARIABLE result OUTPUT_VARIABLE output ERROR_VARIABLE errors)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Example build failed:\n${output}${errors}")
endif()
execute_process(COMMAND "${EXECUTABLE}" OUTPUT_FILE "${ACTUAL}"
    ERROR_VARIABLE errors RESULT_VARIABLE result TIMEOUT 30)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "Example failed (${result}): ${errors}")
endif()
execute_process(COMMAND "${CMAKE_COMMAND}" -E compare_files "${ACTUAL}" "${EXPECTED}"
    RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    file(READ "${EXPECTED}" expected)
    file(READ "${ACTUAL}" actual)
    message(FATAL_ERROR "Example stdout differs byte-for-byte.\nExpected:\n${expected}\nActual:\n${actual}")
endif()
