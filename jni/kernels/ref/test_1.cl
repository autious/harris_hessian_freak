
__kernel void main( global hh_float* input, global hh_float* output )
{
    size_t i = get_global_id(0);
    output[i] = input[i] * input[i];
}
