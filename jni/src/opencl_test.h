#pragma once

#include <stdbool.h>
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS 1
#include <CL/opencl.h>

bool opencl_test_run();
void opencl_test_desaturate_image( const char *input, const char* output );
void opencl_test_save_buffer_to_image( 
    cl_mem buffer, 
    const char *output_filename, 
    unsigned int width, 
    unsigned int height,
    cl_uint num_events_in_wait_list, 
    const cl_event* event_wait_list
    );
cl_mem opencl_test_load_image( 
    const char *input_filename, 
    unsigned int* width, 
    unsigned int* height 
    );
