//__attribute__((work_group_size_hint(32,1,1)))
//__attribute__((reqd_work_group_size(32,1,1)))
//__attribute__((vec_type_hint(hh_float)))
__kernel void gaussx( __global hh_float* gauss_kernel, int kernel_radius, __global hh_float* input, __global hh_float* output, int width, __local hh_float* cached_source )
{
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    int2 local_size = (int2)(get_local_size(0),get_local_size(1));
    int2 local_id = (int2)(get_local_id(0),get_local_id(1));
    int2 group_id = (int2)(get_group_id(0),get_group_id(1));

    int left_index = group_id.x * local_size.x - kernel_radius;

    for( int i = local_id.x; i < kernel_radius + local_size.x + kernel_radius; i += local_size.x )
    {
        cached_source[i] = input[min(width-1,max(left_index+i,0)) + coord.y*width];
    }

    hh_float sum = 0;

    write_mem_fence( CLK_LOCAL_MEM_FENCE );

    for( int i = -kernel_radius; i <= kernel_radius; i++ )
    {
        sum += gauss_kernel[i+kernel_radius] * cached_source[local_id.x + kernel_radius + i];
    }

    output[coord.x+coord.y*width] = sum;
}

__kernel void gaussy( __constant hh_float* gauss_kernel, int kernel_radius, __global hh_float* input, __global hh_float* output, int width, int height)
{
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    int2 local_size = (int2)(get_local_size(0),get_local_size(1));
    int2 local_id = (int2)(get_local_id(0),get_local_id(1));
    int2 group_id = (int2)(get_group_id(0),get_group_id(1));

    int top_index = group_id.y * local_size.y - kernel_radius;
    int left_index = group_id.x * local_size.x - kernel_radius;

    for( int y = local_id.y; y < kernel_radius + local_size.y + kernel_radius; y += local_size.y )
    {
        cached_source[y] = input[min(height-1,max(top_index+y,0))*width + coord.x];
    }

    hh_float sum = 0;

    write_mem_fence( CLK_LOCAL_MEM_FENCE );

    for( int i = -kernel_radius; i <= kernel_radius; i++ )
    {
        sum += gauss_kernel[i+kernel_radius] * cached_source[local_id.y + kernel_radius + i];
    }

    output[coord.x+coord.y*width] = sum;
}

/*
__kernel void gaussx( __global hh_float* gauss_kernel, int kernel_radius, __global hh_float* input, __global hh_float* output, int width, __local hh_float* cached_source )
{
    int2 coord = (int2)(get_global_id(0),get_global_id(1));
    int2 local_size = (int2)(get_local_size(0),get_local_size(1));
    int2 local_id = (int2)(get_local_id(0),get_local_id(1));
    int2 group_id = (int2)(get_group_id(0),get_group_id(1));

    int left_index = group_id.x * local_size.x - kernel_radius;
    int right_index = group_id.x * local_size.x + local_size.x + kernel_radius - 1;

    //get how far into the cache line the data should be offset
    int left_offset = abs(min(left_index, 0));
    int local_len = min(right_index,width-1) - max(0,left_index);

    if( local_len < kernel_radius + local_size.x + kernel_radius )
    {
        for( int i = 0; i < kernel_radius + local_size.x + kernel_radius; i++ )
        {
            cached_source[i] = 0.0f;
        }
    }

    event_t cache_copy_event;
    cache_copy_event = async_work_group_copy( 
        (__local hh_float*)(cached_source + left_offset),
        (__global hh_float*)(&input[max(left_index,0)+coord.y*width]),
        local_len,
        0
    );

    wait_group_events( 1, &cache_copy_event );
    hh_float sum = 0;

    for( int i = -kernel_radius; i <= kernel_radius; i++ )
    {
        sum += gauss_kernel[i+kernel_radius] * cached_source[local_id.x + kernel_radius + i];
    }

    output[coord.x+coord.y*width] = sum;
}
*/

