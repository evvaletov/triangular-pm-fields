cmake_minimum_required(VERSION 3.16)
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 NEW)
endif()
project(triangle_pm_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -fno-fast-math)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-O2)
endif()

find_package(Boost 1.70 REQUIRED)
find_package(Threads REQUIRED)

add_library(triangle_pm STATIC
  src/triangle.cpp
  src/native.cpp
)
target_include_directories(triangle_pm PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${Boost_INCLUDE_DIRS}
)
target_link_libraries(triangle_pm PUBLIC ${CMAKE_DL_LIBS} Threads::Threads)

add_executable(triangle_cli src/triangle_cli.cpp)
target_link_libraries(triangle_cli PRIVATE triangle_pm)

add_executable(native_cli src/native_cli.cpp)
target_link_libraries(native_cli PRIVATE triangle_pm ${CMAKE_DL_LIBS})

add_executable(triangle_example examples/example.cpp)
target_link_libraries(triangle_example PRIVATE triangle_pm)

add_executable(test_triangle tests/test_triangle.cpp)
target_link_libraries(test_triangle PRIVATE triangle_pm ${CMAKE_DL_LIBS})

enable_testing()
add_test(NAME test_triangle COMMAND test_triangle)
if(TARGET tan)
  set_tests_properties(test_triangle PROPERTIES FIXTURES_REQUIRED native_cache_files
    ENVIRONMENT "TRIANGLE_NATIVE_LIB=$<TARGET_FILE:tan>;TRIANGLE_NATIVE_CACHE=${CMAKE_BINARY_DIR}/native_test_work/results/triangle.cache")
endif()
