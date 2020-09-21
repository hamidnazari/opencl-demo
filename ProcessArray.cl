__kernel void ProcessArray(__global int* data, __global int* outData) {
    int gid = get_global_id(0);
    outData[gid] = data[gid] * 2;

    //printf("%d: in: %d out: %d\n", gid, data[gid], outData[gid]);
}
