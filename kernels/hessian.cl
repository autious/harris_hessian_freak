__kernel void hessian( __global float *xx, __global float* xy, __global float* yy, __global float* out, float sigmaD )
{
    int i = get_global_id(0); 
    out[i] = ( xx[i] * yy[i] - pow(xy[i],2) / sigmaD );
}
