cmake_minimum_required(VERSION 3.19)
project(torchplugins VERSION 0.1.0)
message(STATUS "Version: ${PROJECT_VERSION}")

enable_testing()

string(REGEX REPLACE "(.*)/" "" THIS_PACKAGE_NAME "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "This package name is ${THIS_PACKAGE_NAME}")

if (APPLE)
    if (${CMAKE_GENERATOR} MATCHES "Xcode")
            if (${XCODE_VERSION} VERSION_LESS 10)
                message(STATUS "Xcode 10 or higher is required. Please install from the Mac App Store.")
                return ()
            elseif(${XCODE_VERSION} VERSION_GREATER_EQUAL 12)
                set(C74_BUILD_FAT YES)
            endif ()
    endif ()

    if (NOT CMAKE_OSX_ARCHITECTURES)
        if(C74_BUILD_FAT)
            set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "macOS architecture" FORCE)
        else()
            set(CMAKE_OSX_ARCHITECTURES ${CMAKE_SYSTEM_PROCESSOR} CACHE STRING "macOS architecture" FORCE)
        endif()
        message("CMAKE_OSX_ARCHITECTURES set to ${CMAKE_OSX_ARCHITECTURES}")
    endif()
endif()


set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (APPLE)
    set(CMAKE_CXX_FLAGS "-faligned-allocation")
    if (CMAKE_OSX_ARCHITECTURES STREQUAL "")
        set(CMAKE_OSX_ARCHITECTURES ${CMAKE_HOST_SYSTEM_PROCESSOR})
    endif()
    message("CMAKE_OSX_ARCHITECTURES: ${CMAKE_OSX_ARCHITECTURES}")
endif()

# if (LINUX)
#     list(APPEND CMAKE_PREFIX_PATH "/media/fast/libs/libtorch-cxx11-abi-shared-with-deps-1.13.0+cu117/libtorch")
# elseif(APPLE)
#     list(APPEND CMAKE_PREFIX_PATH "/Users/diaz/mambaforge/envs/modal/lib/python3.10/site-packages/torch")
# endif()

# Torch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Max SDK
# include(${CMAKE_CURRENT_SOURCE_DIR}/third_party/min-api/script/min-package.cmake)
set(C74_MIN_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/min-api)

# Add subdirectories
add_subdirectory(src/common/torchwrapper)
add_subdirectory(src/maxmsp/fc)
add_subdirectory(src/maxmsp/encoder)
add_subdirectory(src/maxmsp/filterbank_tilde)