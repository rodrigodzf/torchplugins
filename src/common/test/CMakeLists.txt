set(EXE_NAME test)
set(SRC_FILES model.cpp)

add_executable(${EXE_NAME} ${SRC_FILES})
# set_target_properties(${EXE_NAME} PROPERTIES COMPILE_FLAGS "-m64" LINK_FLAGS "-m64")
target_link_libraries(
    ${PROJECT_NAME}
    PUBLIC
    torchwrapper
)