cmake_minimum_required(VERSION 3.19)
project(torchplugins VERSION 0.1.1)
message(STATUS "Version: ${PROJECT_VERSION}")

# We need C++17 because torch is compiled with C++17 on linux
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_testing()

string(REGEX REPLACE "(.*)/" "" THIS_PACKAGE_NAME "${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "This package name is ${THIS_PACKAGE_NAME}")

if(APPLE)
    if(${CMAKE_GENERATOR} MATCHES "Xcode")
        if(${XCODE_VERSION} VERSION_LESS 10)
            message(STATUS "Xcode 10 or higher is required. Please install from the Mac App Store.")
            return()
        elseif(${XCODE_VERSION} VERSION_GREATER_EQUAL 12)
            set(C74_BUILD_FAT YES)
        endif()
    endif()

    if(NOT CMAKE_OSX_ARCHITECTURES)
        if(C74_BUILD_FAT)
            set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64" CACHE STRING "macOS architecture" FORCE)
        else()
            set(CMAKE_OSX_ARCHITECTURES ${CMAKE_SYSTEM_PROCESSOR} CACHE STRING "macOS architecture" FORCE)
        endif()

        message("CMAKE_OSX_ARCHITECTURES set to ${CMAKE_OSX_ARCHITECTURES}")
    endif()
endif()

if(APPLE)
    set(CMAKE_CXX_FLAGS "-faligned-allocation")

    if(CMAKE_OSX_ARCHITECTURES STREQUAL "")
        set(CMAKE_OSX_ARCHITECTURES ${CMAKE_HOST_SYSTEM_PROCESSOR})
    endif()

    message("CMAKE_OSX_ARCHITECTURES: ${CMAKE_OSX_ARCHITECTURES}")
endif()

# set module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Torch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# FFTW
find_package(FFTW REQUIRED)

# Max SDK
# include(${CMAKE_CURRENT_SOURCE_DIR}/third_party/min-api/script/min-package.cmake)
set(C74_MIN_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/min-api)

# Argparse
# set(FETCHCONTENT_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
include(FetchContent)
option(ARGPARSE_INSTALL "Install argparse" OFF)
FetchContent_Declare(
    argparse
    GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

# Misc setup and subroutines
include(${CMAKE_CURRENT_SOURCE_DIR}/third_party/min-api/script/min-package.cmake)

# Set RPATH
set(CMAKE_MACOSX_RPATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH "@loader_path")

# Add subdirectories
add_subdirectory(src/common/torchwrapper)
add_subdirectory(src/common/test_torchwrapper)
add_subdirectory(src/maxmsp/fc)
add_subdirectory(src/maxmsp/encoder)
add_subdirectory(src/maxmsp/filterbank_tilde)

# Install

# Install the Max package
install(
    DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/externals
    DESTINATION
    ${CMAKE_CURRENT_SOURCE_DIR}/torchplugins
)

# install the extras directory
install(
    DIRECTORY
    ${CMAKE_CURRENT_SOURCE_DIR}/extras
    DESTINATION
    ${CMAKE_CURRENT_SOURCE_DIR}/torchplugins
)

# install the package info, readme and license
install(
    FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/package-info.json
    ${CMAKE_CURRENT_SOURCE_DIR}/README.md
    ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE
    DESTINATION
    ${CMAKE_CURRENT_SOURCE_DIR}/torchplugins
)

# Install the help files
install(
    FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/maxmsp/fc/fc.maxhelp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/maxmsp/encoder/encoder.maxhelp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/maxmsp/filterbank_tilde/filterbank~.maxhelp
    DESTINATION
    ${CMAKE_CURRENT_SOURCE_DIR}/torchplugins/help
)