#pragma once

#include "opencl_handler.h"
#include "freak.h"

void harris_hessian_init( int width, int height );
void harris_hessian_detection( 
    uint8_t *rgba_data, 
    cl_uint event_count, 
    cl_event* event_wait_list, 
    cl_event *event 
);

descriptor* harris_hessian_build_descriptor( 
    int *desc_count,
    cl_uint event_count, 
    cl_event* event_wait_list, 
    cl_event* event 
);

void harris_hessian_close();
