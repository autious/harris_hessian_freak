__kernel void second_moment_matrix_elements( 
    __global float* ddx, 
    __global float* ddy, 
    __global float* xx,
    __global float* xy,
    __global float* yy 
)
{
    int i = get_global_id(0);

    float dx = ddx[i];
    float dy = ddy[i];

    xx[i] = dx*dx;
    xy[i] = dx*dy;
    yy[i] = dy*dy;
}

