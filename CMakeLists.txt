cmake_minimum_required(VERSION 2.8)

project(TGO1D CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "-Wall -g -fPIC")

find_package(NLopt REQUIRED)
find_package(Armadillo REQUIRED)

add_subdirectory(${CMAKE_SOURCE_DIR}/include/fmt EXCLUDE_FROM_ALL)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_subdirectory(test)
add_subdirectory(src)

enable_testing()
add_test(NAME test_all COMMAND unit_tests)

if(APPLE OR UNIX)
  execute_process(COMMAND ln -f -s ${CMAKE_SOURCE_DIR}/test/data WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
else()
  file(COPY ${CMAKE_SOURCE_DIR}/test/data DESTINATION ${PROJECT_BINARY_DIR})
endif()
