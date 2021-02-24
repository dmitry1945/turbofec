#pragma once

int has_even_parity(unsigned int x);
int has_odd_parity(unsigned int x);

#ifdef _MSC_VER
#define API_EXPORT	
#define SSE_ALIGN
#define memalign _aligned_malloc
//#define PARITY(X)	__builtin_parity(X)
//#define PARITY(X)	has_odd_parity(X)
#define PARITY(X)	has_odd_parity(X)
//#define PARITY(X)	(X)

#include < intrin.h >
#  define __builtin_popcount __popcnt

#define POPCNT(X)	__builtin_popcount(X)
#else
#define API_EXPORT	__attribute__((__visibility__("default")))
#define SSE_ALIGN		__attribute__((aligned(16)))
#define PARITY(X)	__builtin_parity(X)
#define POPCNT(X)	__builtin_popcount(X)

#endif


#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
    #define INITIALIZER(f) \
        static void f(void); \
        struct f##_t_ { f##_t_(void) { f(); } }; static f##_t_ f##_; \
        static void f(void)
#elif defined(_MSC_VER)
    #pragma section(".CRT$XCU",read)
    #define INITIALIZER2_(f,p) \
        static void f(void); \
        __declspec(allocate(".CRT$XCU")) void (*f##_)(void) = f; \
        __pragma(comment(linker,"/include:" p #f "_")) \
        static void f(void)
    #ifdef _WIN64
        #define INITIALIZER(f) INITIALIZER2_(f,"")
    #else
        #define INITIALIZER(f) INITIALIZER2_(f,"_")
    #endif
#else
    #define INITIALIZER(f) \
        static void f(void) __attribute__((constructor)); \
        static void f(void)
#endif