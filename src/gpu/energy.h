// Address spaces are part of the type system in Metal. CUDA uses generic
// pointers inside device functions.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
#define GPU_INLINE inline
#define GPU_FN inline
#define GPU_TABLE device
#define GPU_GROUP threadgroup
#define GPU_DP device
#define GPU_METHOD threadgroup
#define GPU_CONSTANT constant
#else
#define GPU_INLINE __device__ __forceinline__
#define GPU_FN __device__
#define GPU_TABLE
#define GPU_GROUP
#define GPU_DP
#define GPU_METHOD
#define GPU_CONSTANT __device__ __constant__
#endif

// Read-only view of the i32 array produced by Params::gpu_tables.
struct Energy {
    GPU_TABLE const int *data;

    enum Offset {
        MlClosing, MlBase, DuplexInit, TerminalAu, Ninio, MaxNinio, MotifCount,
        Stack,
        Hairpin = Stack + 7 * 7,
        Bulge = Hairpin + 129,
        InternalLoop = Bulge + 129,
        MismatchH = InternalLoop + 129,
        MismatchI = MismatchH + 7 * 25,
        Mismatch1n = MismatchI + 7 * 25,
        Mismatch23 = Mismatch1n + 7 * 25,
        Exterior = Mismatch23 + 7 * 25,
        Multibranch = Exterior + 7 * 25,
        Int11 = Multibranch + 7 * 25,
        Int21 = Int11 + 7 * 7 * 25,
        Int22 = Int21 + 7 * 7 * 125,
        InternalLowerBounds = Int22 + 6 * 6 * 256,
        Motifs = InternalLowerBounds + 31
    };

    GPU_INLINE int ml_closing() const { return data[MlClosing]; }
    GPU_INLINE int ml_base() const { return data[MlBase]; }
    GPU_INLINE int duplex_init() const { return data[DuplexInit]; }
    GPU_INLINE int internal_lower_bound(int unpaired) const {
        return data[InternalLowerBounds + unpaired];
    }

    GPU_INLINE int stack(int outer, int inner_rev) const {
        return data[Stack + (outer - 1) * 7 + inner_rev - 1];
    }

    GPU_INLINE int mismatch(int table, int pair, int left, int right) const {
        return data[table + (pair - 1) * 25 + left * 5 + right];
    }

    GPU_INLINE int exterior(int pair, int flank5, int flank3) const {
        return mismatch(Exterior, pair, flank5, flank3);
    }

    GPU_INLINE int multibranch(int pair, int flank5, int flank3) const {
        return mismatch(Multibranch, pair, flank5, flank3);
    }

    GPU_INLINE int end_penalty(int pair) const {
        return pair > 2 ? data[TerminalAu] : 0;
    }

    GPU_INLINE int hairpin(int pair, GPU_GROUP const unsigned char *sequence,
                                          int length) const {
        const int size = length - 2;
        const int cost = data[Hairpin + size];
        if (size < 3) return cost;
        if (size == 3 || size == 4 || size == 6) {
            const int count = data[MotifCount];
            if (count != 0) {
                int key = 1;
                for (int i = 0; i < length; ++i) key = (key << 2) | (sequence[i] - 1);
                for (int m = 0; m < count; ++m) {
                    if (data[Motifs + 2 * m] == key) return data[Motifs + 2 * m + 1];
                }
            }
            if (size == 3) return int(unsigned(cost) + end_penalty(pair));
        }
        return int(unsigned(cost) + mismatch(MismatchH, pair, sequence[1], sequence[size]));
    }

    // si,sj flank the outer pair on the inside. sp,sq flank the inner pair
    // on the outside, and inner_rev describes the pair in reverse order.
    GPU_INLINE int internal(int n1, int n2, int outer, int inner_rev,
                                           int si, int sj, int sp, int sq) const {
        const int small = n1 < n2 ? n1 : n2;
        const int large = n1 > n2 ? n1 : n2;
        const int p = outer - 1;
        const int q = inner_rev - 1;
        if (small == 0) {
            if (large == 0) return stack(outer, inner_rev);
            const unsigned cost = data[Bulge + large];
            if (large == 1) return int(cost + stack(outer, inner_rev));
            return int(cost + end_penalty(outer) + end_penalty(inner_rev));
        }
        if (small == 1 && large == 1)
            return data[Int11 + (p * 7 + q) * 25 + si * 5 + sj];
        if (small == 1 && large == 2) {
            if (n1 == 1) return data[Int21 + (p * 7 + q) * 125 + (si * 5 + sq) * 5 + sj];
            return data[Int21 + (q * 7 + p) * 125 + (sq * 5 + si) * 5 + sp];
        }
        if (small == 2 && large == 2) {
            const int col = (((si - 1) * 4 + sp - 1) * 4 + sq - 1) * 4 + sj - 1;
            return data[Int22 + (p * 6 + q) * 256 + col];
        }
        int table, asymmetry;
        if (small == 2 && large == 3) {
            table = Mismatch23;
            asymmetry = data[Ninio];
        } else {
            table = small == 1 ? Mismatch1n : MismatchI;
            const int penalty = int(unsigned(data[Ninio]) * (large - small));
            asymmetry = penalty < data[MaxNinio] ? penalty : data[MaxNinio];
        }
        return int(unsigned(data[InternalLoop + small + large]) + asymmetry
                   + mismatch(table, outer, si, sj) + mismatch(table, inner_rev, sq, sp));
    }
};
