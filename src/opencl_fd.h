#pragma once 
#include "opencl_handler.h"

struct FD
{
    uint8_t *rgba_host; //Memory of image_rgba_char
    cl_mem image_rgba_char; // "Image source buffer"
    //Following buffers are use when filters are applied to the greyscale image.
    cl_mem buffer_l_float_1; // "From buffer" 
    cl_mem buffer_l_float_2; // "To buffer"
    cl_uint width;
    cl_uint height;
};

bool opencl_fd_load_image( const char *input_filename, struct FD* state );
void opencl_fd_save_buffer_to_image( 
    const char * name, 
    struct FD* state,
    cl_uint num_events_in_wait_list,
    cl_event *event_wait_list );
bool opencl_fd_desaturate_image( 
    struct FD *state,
    cl_uint num_events_in_wait_list,
    cl_event *event_wait_list,
    cl_event *event
);
bool opencl_fd_run_gaussxy( 
    struct FD* state,
    float sigma, 
    cl_uint num_events_in_wait_list, 
    const cl_event* event_wait_list, 
    cl_event* event );
