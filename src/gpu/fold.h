namespace {
enum : int { max_length = 128, infinity = 10000000 };

GPU_FN bool cannot_improve(int stem, int lower, int best) {
#ifdef __METAL_VERSION__
    return addsat(stem, lower) >= best;
#else
    return (long long)stem + lower >= best;
#endif
}

// Three-bit pair types for partners A, C, G, T, in that order.
GPU_CONSTANT unsigned short pair_rows[] = {
    0, 5 << 9, 1 << 6, (2 << 3) | (3 << 9), 6 | (4 << 6)
};

struct FoldState {
    unsigned char seq[max_length + 2];
    unsigned short pair_row[max_length + 2];
    int offset[max_length];          // Diagonal d begins at offset[d].
#ifdef __METAL_VERSION__
    atomic_int min_closed[max_length];
#else
    int min_closed[max_length];
#endif
    GPU_FN int diagonal_min(int d) const GPU_METHOD {
#ifdef __METAL_VERSION__
        return atomic_load_explicit(&min_closed[d], memory_order_relaxed);
#else
        return min_closed[d];
#endif
    }

    int split[3][max_length + 2];    // Closing a multiloop reads diagonal d-2.
    int suffix[max_length + 2];     // Exterior fragments ending at the nick.
    int second[max_length + 2];     // Exterior prefixes of the second strand.
    int whole[max_length + 2];      // Exterior prefixes starting at position 1.

    GPU_FN int pair(int i, int j) const GPU_METHOD {
        return (pair_row[i] >> (3 * (seq[j] - 1))) & 7;
    }

    GPU_FN int index(int i, int j) const GPU_METHOD {
        return offset[j - i] + i - 1;
    }

    GPU_FN int exterior(Energy e, int i, int j, int type, int cut, int n) const GPU_METHOD {
        int left = (i == 1 || i == cut + 1) ? 0 : seq[i - 1];
        int right = (j == n || j == cut) ? 0 : seq[j + 1];
        return e.exterior(type, left, right);
    }

    // At distance d, all enclosed pairs with u unpaired nucleotides lie on
    // diagonal d-u-2. Lanes in a warp read adjacent cells for the same arms.
    GPU_FN int internal(Energy e, GPU_DP const int *closed, int i, int j, int cut,
                            int type, int best) const GPU_METHOD {
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
            if (bounded && cannot_improve(diagonal_min(inner_d), lower, best))
                continue;
            int first = crosses ? max(0, cut + 2 + u - j) : 0;
            int last = crosses ? min(u, cut - i - 1) : u;
            GPU_DP const int *diagonal = closed + offset[inner_d] + i;
            for (int a = first; a <= last; ++a) {
                int inner_energy = diagonal[a];
                if (inner_energy < infinity && (!bounded || !cannot_improve(inner_energy, lower, best))) {
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

    GPU_FN int closed_pair(Energy e, GPU_DP const int *closed, int i, int j, int cut,
                               int type) const GPU_METHOD {
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

    GPU_FN int prefix(Energy e, GPU_DP const int *closed, int start, int j, int cut,
                          int n, GPU_GROUP const int *previous) const GPU_METHOD {
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
GPU_FN void fold_barrier() {
#ifdef __METAL_VERSION__
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
#else
    __syncthreads();
#endif
}

// All lanes in the group enter together and own one ordered pair.
GPU_FN void fill_pair(GPU_TABLE const int *params, GPU_TABLE const unsigned char *input,
                      int n, int cut, GPU_GROUP FoldState &state, GPU_DP int *closed,
                      GPU_DP int *branch, int lane, int width) {
    Energy e{params};
    for (int t = lane; t < n + 2; t += width) {
        state.seq[t] = input[t];
        state.pair_row[t] = pair_rows[input[t]];
        state.suffix[t] = state.second[t] = state.whole[t] = 0;
        state.split[0][t] = infinity;
        if (t < n) {
            state.offset[t] = t * (2 * n - t + 1) / 2;
#ifdef __METAL_VERSION__
            atomic_store_explicit(&state.min_closed[t], int(infinity), memory_order_relaxed);
#else
            state.min_closed[t] = infinity;
#endif
            closed[t] = branch[t] = infinity;
        }
    }
    fold_barrier();

    for (int d = 1; d < n; ++d) {
        for (int i = lane + 1; i <= n - d; i += width) {
            int j = i + d;
            int type = state.pair(i, j);
            int stem = type ? state.closed_pair(e, closed, i, j, cut, type) : infinity;
            int cell = state.offset[d] + i - 1;
            closed[cell] = stem;
            if (stem < infinity)
#ifdef __METAL_VERSION__
                atomic_fetch_min_explicit(&state.min_closed[d], stem, memory_order_relaxed);
#else
                atomicMin(&state.min_closed[d], stem);
#endif

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
        fold_barrier();
    }
}
}  // namespace
