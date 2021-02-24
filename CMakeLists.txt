project(turbofec_project)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 14)

set ( SRC_DIR ${PROJECT_SOURCE_DIR}/src )


include_directories (${PROJECT_SOURCE_DIR}/include)

add_subdirectory(src)

add_subdirectory(tests)

