#ifndef HARRIS_HESSIAN_FREAK_H
#define HARRIS_HESSIAN_FREAK_H

#include <CL/opencl.h>

typedef uint8_t word_t;

struct keyPoint_t {
	int x; // pixel coordinates
	int y;
	float size; // gaussian sigma
}; typedef struct keyPoint_t keyPoint;

struct descriptor_t {
	word_t* data;
	uint32_t x;
	uint32_t y;
}; typedef struct descriptor_t descriptor;

void harris_hessian_freak_init( int width, int height );
void harris_hessian_freak_detection( 
    uint8_t *rgba_data, 
    const char* save_path,
    cl_uint event_count, 
    cl_event* event_wait_list, 
    cl_event *event 
);

descriptor* harris_hessian_freak_build_descriptor( 
    keyPoint* keypoints,
    size_t keypoint_count,  
    size_t *desc_count,
    cl_uint event_count, 
    cl_event* event_wait_list, 
    cl_event* event 
);

keyPoint* harris_hessian_freak_generate_keypoint_list( 
    size_t* in_size, 
    cl_uint event_count, 
    cl_event *event_wait_list, 
    cl_event* event 
);

void harris_hessian_freak_close();

#endif
