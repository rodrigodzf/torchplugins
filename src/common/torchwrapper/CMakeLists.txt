set(LIB_NAME torchwrapper)
set(SRC_FILES model.cpp)

add_library(${LIB_NAME} SHARED ${SRC_FILES})
target_include_directories(${LIB_NAME} PUBLIC "include")
# set_target_properties(${LIB_NAME} PROPERTIES COMPILE_FLAGS "-m64" LINK_FLAGS "-m64")
target_link_libraries(${LIB_NAME} PUBLIC ${TORCH_LIBRARIES})

