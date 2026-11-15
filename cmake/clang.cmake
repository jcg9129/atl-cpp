# Pin Clang's libstdc++ selection too, even on hosts with a newer GCC installed.
find_program(ATL_GCC_DRIVER g++-15 REQUIRED)
execute_process(COMMAND "${ATL_GCC_DRIVER}" -print-libgcc-file-name
    OUTPUT_VARIABLE gcc_runtime OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
get_filename_component(gcc_install_dir "${gcc_runtime}" DIRECTORY)
set(CMAKE_CXX_FLAGS_INIT "--gcc-install-dir=${gcc_install_dir}")
