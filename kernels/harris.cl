
__constant float alpha = 0.04; // I have found literally one lecture that explains that this alpha is 
                   //empirically measured to be [0.04 - 0.06] and another that just uses 
                   //0.04 without ref
__kernel void harris_corner_response( 
        __global float* xx, 
        __global float* xy, 
        __global float* yy, 
        __global float* output, 
        float sigmaD )
{
    int i = get_global_id(0);
    float A = xx[i] * pow(sigmaD, 2);
    float B = yy[i] * pow(sigmaD, 2);
    float C = xy[i] * pow(sigmaD, 2);

    output[i] = A * B - C * C - alpha * pow(A + B, 2);
}
