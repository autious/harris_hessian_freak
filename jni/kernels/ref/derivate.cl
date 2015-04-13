__kernel void derivate( __global hh_float* input, __global hh_float* ddx, __global hh_float* ddy, int width, int height, int sample_width )
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    hh_float v0, v1;

    v0 = input[width * y + il(x-sample_width,0,width-1)];
    v1 = input[width * y + il(x+sample_width,0,width-1)];
    ddx[x+y*width] = v0 - v1;

    v0 = input[width * il(y+sample_width,0,height-1) + x];
    v1 = input[width * il(y-sample_width,0,height-1) + x];
    ddy[x+y*width] = v0 - v1;
}

__kernel void derivate_y( __global hh_float* input, __global hh_float* ddy, int width, int height, int sample_width )
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    hh_float v0, v1;

    v0 = input[width * il(y+sample_width,0,height-1) + x];
    v1 = input[width * il(y-sample_width,0,height-1) + x];
    ddy[x+y*width] = v0 - v1;
}

__kernel void derivate_x( __global hh_float* input, __global hh_float* ddx, int width, int height, int sample_width )
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    hh_float v0, v1;

    v0 = input[width * y + il(x-sample_width,0,width-1)];
    v1 = input[width * y + il(x+sample_width,0,width-1)];
    ddx[x+y*width] = v0 - v1;
}
