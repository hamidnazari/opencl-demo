__kernel void NumericalReduction(__global int* data, __local int* localData, __global int* outData) {
    size_t globalId = get_global_id(0);
    size_t localSize = get_local_size(0);
    size_t localId = get_local_id(0);
    size_t groupId = get_group_id(0);

    // printf("globalId: %d, localSize: %d, localId: %d, groupId: %d, value: %d\n",
    //        globalId, localSize, localId, groupId, data[globalId]);

    localData[localId] = data[globalId];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = localSize >> 1; i > 0; i >>= 1) {
        if (localId < i) {
            // printf("localId: %d, i: %d, localId+i: %d, %d+%d=%d\n",
            //         localId, i, localId + i,
            //         localData[localId], localData[localId + i], localData[localId] + localData[localId + i]);
            localData[localId] += localData[localId + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId == 0) {
        outData[groupId] = localData[0];
    }
}
