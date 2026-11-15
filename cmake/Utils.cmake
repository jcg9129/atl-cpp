function(target_name_for out_target out_file file)
    get_filename_component(_extension "${file}" LAST_EXT)

    file(RELATIVE_PATH _relative ${PROJECT_SOURCE_DIR} ${file})
    string(REPLACE "${_extension}" "" _name ${_relative})
    string(REGEX REPLACE "/" "-" _name ${_name})
    set(${out_target}
        "${ATL_TARGET_PREFIX}${_name}"
        PARENT_SCOPE)

    file(RELATIVE_PATH _relative ${CMAKE_CURRENT_LIST_DIR} ${file})
    string(REPLACE "${_extension}" "" _name ${_relative})
    string(REGEX REPLACE "/" "-" _name ${_name})
    set(${out_file}
        "${_name}"
        PARENT_SCOPE)
endfunction()
