#pragma once

#include <stdbool.h>
#include <stdio.h>

#include <CL/opencl.h>

extern bool opencl_timer_enable_profile;

void opencl_timer_push_event( const char* name, cl_event event );
void opencl_timer_push_marker( const char* name, int reoccurance );
void opencl_timer_print_results( FILE* f );
void opencl_timer_clear_events( );
