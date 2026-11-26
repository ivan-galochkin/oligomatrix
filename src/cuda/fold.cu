#include "energy.cuh"

namespace {
constexpr int max_length = 128;
constexpr int shared_branch_limit = 96;
constexpr int infinity = 10000000;

// Three-bit pair types for partners A, C, G, T, in that order.
__device__ __constant__ unsigned short pair_rows[] = {
    0, 5 << 9, 1 << 6, (2 << 3) | (3 << 9), 6 | (4 << 6)
};

struct FoldState {
    unsigned char seq[max_length + 2];
    unsigned short pair_row[max_length + 2];
    int offset[max_length];          // Diagonal d begins at offset[d].
    int min_closed[max_length];      // Minimum feasible stem on each finished diagonal.
    int split[3][max_length + 2];    // Closing a multiloop reads diagonal d-2.
    int suffix[max_length + 2];     // Exterior fragments ending at the nick.
    int second[max_length + 2];     // Exterior prefixes of the second strand.
    int whole[max_length + 2];      // Exterior prefixes starting at position 1.

    __device__ int pair(int i, int j) const {
        return (pair_row[i] >> (3 * (seq[j] - 1))) & 7;
    }

    __device__ int index(int i, int j) const {
        return offset[j - i] + i - 1;
    }

    __device__ int exterior(Energy e, int i, int j, int type, int cut, int n) const {
        int left = (i == 1 || i == cut + 1) ? 0 : seq[i - 1];
        int right = (j == n || j == cut) ? 0 : seq[j + 1];
        return e.exterior(type, left, right);
    }

    // At distance d, all enclosed pairs with u unpaired nucleotides lie on
    // diagonal d-u-2. Lanes in a warp read adjacent cells for the same arms.
    __device__ int internal(Energy e, const int *closed, int i, int j, int cut,
                            int type, int best) const {
        int d = j - i;
        bool crosses = i <= cut && cut < j;
        int si = seq[i + 1];
        int sj = seq[j - 1];
        int max_unpaired = min(30, d - 3);
        for (int u = 0; u <= max_unpaired; ++u) {
            int inner_d = d - u - 2;
            int lower = e.internal_lower_bound(u);
            bool bounded = lower != (-2147483647 - 1);
            // Every candidate on this diagonal costs at least its cheapest
            // stem plus the cheapest loop with u unpaired bases. The host
            // disables this bound for tables that can overflow local energy.
            if (bounded && (long long)min_closed[inner_d] + lower >= best)
                continue;
            int first = crosses ? max(0, cut + 2 + u - j) : 0;
            int last = crosses ? min(u, cut - i - 1) : u;
            const int *diagonal = closed + offset[inner_d] + i;
            for (int a = first; a <= last; ++a) {
                int inner_energy = diagonal[a];
                if (inner_energy < infinity && (!bounded || (long long)inner_energy + lower < best)) {
                    int k = i + 1 + a;
                    int l = j - 1 - (u - a);
                    int local = e.internal(a, u - a, type, pair(l, k),
                                           si, sj, seq[k - 1], seq[l + 1]);
                    best = min(best, inner_energy + local);
                }
            }
        }
        return best;
    }

    __device__ int closed_pair(Energy e, const int *closed, int i, int j, int cut,
                               int type) const {
        int d = j - i;
        int reverse = ((type - 1) ^ 1) + 1;
        int best;
        if (i <= cut && cut < j) {
            int left = j == cut + 1 ? 0 : seq[j - 1];
            int right = i == cut ? 0 : seq[i + 1];
            best = e.duplex_init() + e.exterior(reverse, left, right)
                 + suffix[i + 1] + second[j - 1];
        } else {
            best = e.hairpin(type, seq + i, d + 1);
        }
        if (d >= 2 && i != cut && j != cut + 1) {
            int inside = split[(d - 2) % 3][i + 1];
            if (inside < infinity) {
                best = min(best, inside + e.ml_closing()
                                 + e.multibranch(reverse, seq[j - 1], seq[i + 1]));
            }
        }
        return internal(e, closed, i, j, cut, type, best);
    }

    __device__ int prefix(Energy e, const int *closed, int start, int j, int cut,
                          int n, const int *previous) const {
        int best = j == cut + 1 ? infinity : previous[j - 1];
        for (int i = start; i < j; ++i) {
            if ((i == cut + 1 && i != start) || previous[i - 1] == infinity)
                continue;
            int stem = closed[index(i, j)];
            if (stem < infinity) {
                best = min(best, previous[i - 1] + stem + exterior(e, i, j, pair(i, j), cut, n));
            }
        }
        return best;
    }
};
}  // namespace

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
    int *branch = capacity <= shared_branch_limit ? storage + triangle
                                                  : scratch + (size_t)job * triangle;
    Energy e{params};
    const unsigned char *input = seqs + (size_t)job * (capacity + 2);
    for (int t = lane; t < n + 2; t += blockDim.x) {
        state.seq[t] = input[t];
        state.pair_row[t] = pair_rows[input[t]];
        state.suffix[t] = state.second[t] = state.whole[t] = 0;
        state.split[0][t] = infinity;
        if (t < n) {
            state.offset[t] = t * (2 * n - t + 1) / 2;
            state.min_closed[t] = infinity;
            closed[t] = branch[t] = infinity;
        }
    }
    __syncthreads();

    for (int d = 1; d < n; ++d) {
        for (int i = lane + 1; i <= n - d; i += blockDim.x) {
            int j = i + d;
            int type = state.pair(i, j);
            int stem = type ? state.closed_pair(e, closed, i, j, cut, type) : infinity;
            int cell = state.offset[d] + i - 1;
            closed[cell] = stem;
            if (stem < infinity)
                atomicMin(&state.min_closed[d], stem);

            int split = infinity;
            // The right branch must contain at least three positions, and a
            // split at the nick would disconnect the loop.
            for (int k = i + 1; k < j - 2; ++k) {
                int left = branch[state.index(i, k)];
                int right = branch[state.index(k + 1, j)];
                if (k != cut && left < infinity && right < infinity)
                    split = min(split, left + right);
            }
            state.split[d % 3][i] = split;
            int best = split;
            if (i != cut + 1 && j != cut) {
                if (stem < infinity)
                    best = min(best, stem + e.multibranch(type, state.seq[i - 1], state.seq[j + 1]));
                int left = branch[state.index(i + 1, j)];
                int right = branch[state.index(i, j - 1)];
                if (i != cut && left < infinity)
                    best = min(best, left + e.ml_base());
                if (j != cut + 1 && right < infinity)
                    best = min(best, right + e.ml_base());
            }
            branch[cell] = best;

            // The only same-diagonal stem in each exterior scan is the cell
            // just written by this lane. Everything else is on an earlier one.
            if (j == cut) {
                int suffix = state.suffix[i + 1];
                for (int k = i + 1; k <= cut; ++k) {
                    int inner = closed[state.index(i, k)];
                    if (inner < infinity) {
                        suffix = min(suffix, inner + state.exterior(e, i, k, state.pair(i, k), cut, n)
                                                   + state.suffix[k + 1]);
                    }
                }
                state.suffix[i] = suffix;
            }
            if (i == cut + 1)
                state.second[j] = state.prefix(e, closed, i, j, cut, n, state.second);
            if (i == 1)
                state.whole[j] = state.prefix(e, closed, i, j, cut, n, state.whole);
        }
        __syncthreads();
    }
    if (lane == 0) {
        float connected = __fdiv_rn((float)state.whole[n], 100.0f);
        if (cut <= n) {
            float separate = __fdiv_rn((float)state.whole[cut], 100.0f) + monomers[job];
            connected = fminf(connected, separate);
        }
        out[job] = connected;
    }
}
