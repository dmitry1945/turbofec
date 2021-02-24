project(turbofec_lib)
cmake_minimum_required(VERSION 3.0)

set(CMAKE_CXX_STANDARD 14)

set ( SRC_DIR . )


include_directories (../include)

add_definitions(-DHAVE_SSE3)

list(APPEND SRC 
    conv_dec.cpp
    conv_enc.cpp
    conv_rate_match.cpp
    turbo_dec.cpp
    turbo_enc.cpp
    turbo_rate_match.cpp
)
# add_executable ( ${APP_NAME} ${SRC_DIR}/conv_dec.c )

# add_executable ( ${APP_NAME} ${SRC} )

#add_library(turbofec_lib SHARED  ${SRC})
add_library(turbofec_lib STATIC  ${SRC})