#include "energy.h"
#include "fold.h"

// A block owns one sequence pair. The two triangles use 32-bit centi-kcal/mol
// throughout. For larger capacities the branch triangle uses global scratch.
extern "C" __global__ void fold(const int *params, const unsigned char *seqs,
                                const int *cuts, const int *lengths, const float *monomers,
                                float *out, int *scratch, int capacity) {
    int job = blockIdx.x;
    int n = lengths[job];
    int cut = cuts[job];
    int lane = threadIdx.x;
    if (n == 1 || (n == 2 && cut == 1)) {
        if (lane == 0)
            out[job] = 0.0f;
        return;
    }

    __shared__ FoldState state;
    extern __shared__ int storage[];
    int triangle = capacity * (capacity + 1) / 2;
    int *closed = storage;
    int *branch = capacity <= 96 ? storage + triangle
                                                  : scratch + (size_t)job * triangle;
    fill_pair(params, seqs + (size_t)job * (capacity + 2), n, cut, state, closed, branch, lane, blockDim.x);
    if (lane == 0) {
        float connected = __fdiv_rn((float)state.whole[n], 100.0f);
        if (cut <= n) {
            float separate = __fdiv_rn((float)state.whole[cut], 100.0f) + monomers[job];
            connected = fminf(connected, separate);
        }
        out[job] = connected;
    }
}
