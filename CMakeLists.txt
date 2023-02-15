cmake_minimum_required(VERSION 3.11)
project(opencl)

option(INSOURCEBUILD OFF)

if((${PROJECT_SOURCE_DIR} STREQUAL ${PROJECT_BINARY_DIR}) AND NOT ${INSOURCEBUILD})
  message(FATAL_ERROR "In-source building disabled. Provide -DINSOURCEBUILD option if you are sure about it.")
endif()

set(CMAKE_CXX_STANDARD 20)
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libstdc++")
endif()

# Possibly support windows and MSVC compiler
if(MSVC)
  add_compile_options(/W3)
else()
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(GTest)
set(NOGTEST OFF CACHE BOOL "Disable GoogleTest")

enable_testing()
if(NOT NOGTEST AND GTEST_FOUND AND CMAKE_MINOR_VERSION GREATER_EQUAL 20)
  set(ENABLE_GTEST ON)
else()
  message(WARNING "Google Test disabled")
endif()

set(NOLINT ON CACHE BOOL "Disable clang-tidy")

if(NOT ${NOLINT})
find_program(CLANG_TIDY_COMMAND clang-tidy)
if(CLANG_TIDY_COMMAND)

# Clang-tidy for linting
set(CMAKE_CXX_CLANG_TIDY 
  clang-tidy;
  -checks=google-*,cppcoreguidelines-*
)
set(CMAKE_C_CLANG_TIDY 
  clang-tidy;
  -checks=google-*,cppcoreguidelines-*
)

else()
message(WARNING "Unable to find clang-tidy, linter disabled")
endif()

else()
message(WARNING "-DNOLINT option provided, linter disabled")
endif()

set(DCMAKE_EXPORT_COMPILE_COMMANDS ON)

find_program(BASH_PROGRAM bash)

# Add flags for GNU sampling profiler gprof in Debug build and RelWithDebInfo
option(PROFILE OFF)
if (PROFILE)
  add_compile_options(-pg)
  add_link_options(-pg)
endif()

option(SANITIZE OFF)
if (SANITIZE)
  add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=undefined -fno-omit-frame-pointer)
endif()

add_subdirectory(3rd-party/OpenCL-Headers)
add_subdirectory(3rd-party/OpenCL-ICD-Loader)
add_subdirectory(3rd-party/OpenCL-CLHPP)

set(RESISTORS_NO_TESTING__ ON CACHE BOOL "Disable testing" FORCE)
add_subdirectory(linmath/lib EXCLUDE_FROM_ALL)

find_package(Boost REQUIRED COMPONENTS program_options)

set(COMMON_INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/common
)

set (INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

function(add_opencl_program TARGET_NAME INPUT_FILES OPENCL_VERSION)
  add_executable(${TARGET_NAME} ${INPUT_FILES})
  target_link_libraries(${TARGET_NAME} PUBLIC OpenCL::OpenCL OpenCL::Headers OpenCL::HeadersCpp Boost::program_options)
  target_compile_definitions(${TARGET_NAME} PUBLIC CL_HPP_TARGET_OPENCL_VERSION=${OPENCL_VERSION} CL_TARGET_OPENCL_VERSION=${OPENCL_VERSION})
  target_include_directories(${TARGET_NAME} PUBLIC ${COMMON_INCLUDE_DIR})
  target_include_directories(${TARGET_NAME} PUBLIC ${INCLUDE_DIR})
endfunction()

add_opencl_program(oclinfo oclinfo.cc 220)
add_opencl_program(vadd vadd.cc 220)

add_opencl_program(matmult matmult.cc 220)
target_link_libraries(matmult PUBLIC throttle)

add_opencl_program(bitonic bitonic.cc 220)