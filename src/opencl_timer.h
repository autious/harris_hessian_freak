#pragma once

#ifdef PROFILE

#include <stdbool.h>
#include <stdio.h>

#include <CL/opencl.h>

extern bool opencl_timer_enable_profile;

void opencl_timer_push_event( const char* name, cl_event event );
int opencl_timer_push_marker( const char* name, int reoccurance );
void opencl_timer_push_segment( const char* name, int start, int end );
void opencl_timer_print_results( FILE* f );
void opencl_timer_clear_events( );

#define PROFILE_PE(kernel,event) opencl_timer_push_event(#kernel,event)
#define PROFILE_PM(kernel,o) opencl_timer_push_marker(#kernel,o)
#else
#define PROFILE_PE(kernel,event) ((void)0)
#define PROFILE_PM(kernel,o) ((int)0)
#endif
