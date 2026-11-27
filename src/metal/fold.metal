// energy.h and fold.h precede this entry point in the compiled source.
// Store the triangular tables in device memory so a 128-base pair also fits
// devices with a 32 KiB threadgroup-memory limit.
kernel void fold(device const int *params [[buffer(0)]],
                 device const uchar *sequences [[buffer(1)]],
                 device const int2 *sizes [[buffer(2)]],
                 device int2 *energies [[buffer(3)]],
                 device int *scratch [[buffer(4)]],
                 constant int &capacity [[buffer(5)]],
                 uint job [[threadgroup_position_in_grid]],
                 uint lane [[thread_index_in_threadgroup]],
                 uint group_size [[threads_per_threadgroup]]) {
    int cut = sizes[job].x;
    int n = sizes[job].y;
    if (n == 2) {
        if (lane == 0) energies[job] = int2(0);
        return;
    }
    threadgroup FoldState state;
    uint triangle = capacity * (capacity + 1) / 2;
    device int *closed = scratch + job * 2 * triangle;
    fill_pair(params, sequences + job * (capacity + 2), n, cut, state,
              closed, closed + triangle, int(lane), int(group_size));
    if (lane == 0) energies[job] = int2(state.whole[n], state.whole[cut]);
}
