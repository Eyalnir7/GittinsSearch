#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>

static __thread int depth = 0;

extern "C" void __cyg_profile_func_enter(void* func, void* caller)
    __attribute__((no_instrument_function));

extern "C" void __cyg_profile_func_exit(void* func, void* caller)
    __attribute__((no_instrument_function));

static void print_sym(void* f) __attribute__((no_instrument_function));

static void print_sym(void* f)
{
    Dl_info info;
    if (dladdr(f, &info) && info.dli_sname)
        fprintf(stderr, "%s", info.dli_sname);
    else
        fprintf(stderr, "%p", f);
}

extern "C" void __cyg_profile_func_enter(void* func, void*)
{
    for (int i = 0; i < depth; i++) write(2, " ", 1);
    write(2, ">> ", 3);
    print_sym(func);
    write(2, "\n", 1);
    depth++;
}

extern "C" void __cyg_profile_func_exit(void* func, void*)
{
    depth--;
    for (int i = 0; i < depth; i++) write(2, " ", 1);
    write(2, "<< ", 3);
    print_sym(func);
    write(2, "\n", 1);
}