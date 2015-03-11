
#define il(value,v_min,v_max) min(max(v_min,value),v_max)

__kernel void derivate( __global float* input, __global float* ddx, __global float* ddy, int width, int height )
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    float v0, v1;

    v0 = input[width * y + il(x-1,0,width-1)];
    v1 = input[width * y + il(x+1,0,width-1)];
    ddx[x+y*width] = v0 - v1;

    v0 = input[width * il(y + 1,0,height-1) + x];
    v1 = input[width * il(y - 1,0,height-1) + x];
    ddy[x+y*width] = v0 - v1;
}
