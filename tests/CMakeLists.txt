project(turbofec_test)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 17)

set ( SRC_DIR ${PROJECT_SOURCE_DIR}/src )

add_definitions(-DHAVE_SSE3)
include_directories (${PROJECT_SOURCE_DIR}/include)
include_directories(../srs)
include_directories(${CMAKE_CURRENT_LIST_DIR})

message(test - "${CMAKE_CURRENT_LIST_DIR}")

list(APPEND SRC  main
    turbo_test.cpp 
    noise.cpp
    optargs.cpp
)

LINK_DIRECTORIES("${CMAKE_BINARY_DIR}/src")

add_executable( ${APP_NAME} ${SRC} )

# add_library(turbofec_lib STATIC IMPORTED)

target_link_libraries( ${APP_NAME} 
    turbofec_lib
)