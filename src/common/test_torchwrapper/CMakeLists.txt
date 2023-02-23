set(EXE_NAME test_torchwrapper)
set(SRC_FILES main.cpp)

add_executable(${EXE_NAME} ${SRC_FILES})
# set_target_properties(${EXE_NAME} PROPERTIES COMPILE_FLAGS "-m64" LINK_FLAGS "-m64")
target_link_libraries(
    ${EXE_NAME}
    PUBLIC
    torchwrapper
    argparse
)
set_target_properties(${EXE_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
