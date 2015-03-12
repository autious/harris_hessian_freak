__kernel void harrisCornerResponse( 
        __global float* xx, 
        __global float* xy, 
        __global float* yy, 
        __global float* output, 
        int width, 
        int height,
        float sigmaD )
{
    float A = xx[i] * pow(sigmaD, 2);
    float B = yy[i] * pow(sigmaD, 2);
    float C = xy[i] * pow(sigmaD, 2);

    output[i] = A * B - C * C - alpha * pow(A + B, 2);
}
