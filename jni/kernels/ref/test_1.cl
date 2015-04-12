
__kernel void main( global hh_float* input, global hh_float* output )
{
    size_t i = get_global_id(0);
    STORE_HHF(output, i, LOAD_HHF(input,i) * LOAD_HHF(input,i));
}
