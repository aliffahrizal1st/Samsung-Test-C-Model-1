cmake_minimum_required(VERSION 3.10)
set(CMAKE_TOOLCHAIN_FILE "D:/Projects/C++ Utils/vcpkg/scripts/buildsystems/vcpkg.cmake" CACHE STRING "")

project(Samsung-Test-C-Model-1)

# Find dependencies
find_package(fmt CONFIG REQUIRED)
find_package(frugally-deep CONFIG REQUIRED)

# Add executable
add_executable(TextClassification Main.cpp)

# Link libraries
target_link_libraries(TextClassification PRIVATE fmt::fmt-header-only)
target_link_libraries(TextClassification PRIVATE frugally-deep::fdeep)

