include(${C74_MIN_API_DIR}/script/min-pretarget.cmake)

include_directories( 
	"${C74_INCLUDES}"
)

set( SOURCE_FILES
	${PROJECT_NAME}.cpp
)

add_library( 
	${PROJECT_NAME} 
	MODULE
	${SOURCE_FILES}
)

target_compile_definitions(
    ${PROJECT_NAME}
    PUBLIC
    VERSION="${CMAKE_PROJECT_VERSION}"
)

target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
    torchwrapper
)

include(${C74_MIN_API_DIR}/script/min-posttarget.cmake)

if (APPLE)
    add_custom_command( 
            TARGET ${PROJECT_NAME} 
            POST_BUILD 
            COMMAND cp "${TORCH_INSTALL_PREFIX}/lib/*.dylib" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}.mxo/Contents/MacOS/" 
            COMMENT "Copy Torch Libraries" 
        )
    
    if (CMAKE_OSX_ARCHITECTURES STREQUAL "arm64")
        add_custom_command( 
            TARGET ${PROJECT_NAME} 
            POST_BUILD 
            COMMAND "codesign" "--force" "--deep" "-s" "-" "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/${${PROJECT_NAME}_EXTERN_OUTPUT_NAME}.mxo"
            COMMENT "Codesign external" 
        )
    endif()
endif()

#############################################################
# UNIT TEST
#############################################################

include(${C74_MIN_API_DIR}/test/min-object-unittest.cmake)
