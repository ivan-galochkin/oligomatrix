//! Minimum free energies for linear one- and two-strand fold compounds.
//! Positions are one-based. A nick after position `cut` separates two strands.

use crate::params::Params;

const INF: i32 = 10_000_000;
const MAX_INTERIOR: usize = 30;
const PAIR: [[u8; 5]; 5] = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 5],
    [0, 0, 0, 1, 0],
    [0, 0, 2, 0, 3],
    [0, 6, 0, 4, 0],
];
const REVERSE: [u8; 7] = [0, 2, 1, 4, 3, 6, 5];

#[derive(Clone, Copy)]
struct Stem {
    energy: i32,
    left: u16,
    pair: u8,
}

struct Fold<'a> {
    p: &'a Params,
    seq: Vec<u8>,
    n: usize,
    cut: usize,
    // Only feasible enclosed pairs, grouped by right endpoint.
    stems: Vec<Vec<Stem>>,
}

impl<'a> Fold<'a> {
    fn new(first: &[u8], second: &[u8], p: &'a Params) -> Self {
        let n = first.len() + second.len();
        let cut = if second.is_empty() {
            n + 1
        } else {
            first.len()
        };
        let mut seq = Vec::with_capacity(n + 2);
        seq.push(0);
        seq.extend(first.iter().chain(second).map(|base| match base {
            b'A' => 1,
            b'C' => 2,
            b'G' => 3,
            b'T' => 4,
            _ => unreachable!("caller validates fragments"),
        }));
        seq.push(seq[1]);
        seq[0] = seq[n];
        Self {
            p,
            seq,
            n,
            cut,
            stems: vec![Vec::new(); n + 1],
        }
    }

    fn exterior(&self, i: usize, j: usize, pair: u8) -> i32 {
        let left = if i == 1 || i == self.cut + 1 {
            0
        } else {
            self.seq[i - 1]
        };
        let right = if j == self.n || j == self.cut {
            0
        } else {
            self.seq[j + 1]
        };
        self.p.exterior(pair, left, right)
    }

    /// Exterior prefixes rooted at the first nucleotide of a strand. Extending
    /// across the nick requires a stem that actually connects the two strands.
    fn prefix(&self, start: usize, end: usize) -> Vec<i32> {
        let mut prefix = vec![0; end + 1];
        for j in start + 1..=end {
            let mut best = if j == self.cut + 1 {
                INF
            } else {
                prefix[j - 1]
            };
            for stem in &self.stems[j] {
                let i = usize::from(stem.left);
                if i < start || (i == self.cut + 1 && i != start) || prefix[i - 1] == INF {
                    continue;
                }
                best = best.min(prefix[i - 1] + stem.energy + self.exterior(i, j, stem.pair));
            }
            prefix[j] = best;
        }
        prefix
    }

    /// Try every inner pair with at most MAX_INTERIOR unpaired bases.
    /// Neither arm of an internal loop may pass the nick.
    fn internal(&self, i: usize, j: usize, pair: u8, mut best: i32) -> i32 {
        let crosses = i <= self.cut && self.cut < j;
        let min_right = if crosses { self.cut + 1 } else { 0 };
        let max_left = if crosses { self.cut } else { j };
        let first = (i + 2)
            .max(j.saturating_sub(MAX_INTERIOR + 1))
            .max(min_right);
        for l in first..j {
            let n2 = j - l - 1;
            let last = (i + 1 + MAX_INTERIOR - n2).min(max_left);
            // Columns were filled from right to left, so reversing them visits
            // the nearest inner pairs first and stops when the loop is too big.
            for inner in self.stems[l].iter().rev() {
                let k = usize::from(inner.left);
                if k > last {
                    break;
                }
                let local = self.p.internal(
                    k - i - 1,
                    n2,
                    pair,
                    REVERSE[inner.pair as usize],
                    self.seq[i + 1],
                    self.seq[j - 1],
                    self.seq[k - 1],
                    self.seq[l + 1],
                );
                best = best.min(inner.energy + local);
            }
        }
        best
    }

    fn fill(&mut self) -> Vec<i32> {
        let n = self.n;
        // Column j of the triangular table holds M[i,j], at least one
        // multiloop branch. Its current row is also contiguous for the splits.
        let offsets: Vec<usize> = (0..=n).map(|j| j * j.saturating_sub(1) / 2).collect();
        let mut branches = vec![INF; offsets[n] + n + 1];
        let mut row = vec![INF; n + 1];
        let mut prev_split = vec![INF; n + 1];
        let mut curr_split = vec![INF; n + 1];
        let mut new_stems = Vec::with_capacity(n);

        // Exterior fragments on the inside of the innermost intermolecular
        // pair run from i+1 to the nick, and from the nick to j-1.
        let mut suffix = vec![0; n + 2];
        let mut second_prefix = Vec::new();
        for i in (1..n).rev() {
            if i == self.cut {
                second_prefix = self.prefix(self.cut + 1, n);
            }
            let mut first_suffix = suffix[i + 1];
            for j in i + 1..=n {
                let pair = PAIR[self.seq[i] as usize][self.seq[j] as usize];
                let mut closed = INF;
                if pair != 0 {
                    let rev = REVERSE[pair as usize];
                    closed = if i <= self.cut && self.cut < j {
                        let left = if j == self.cut + 1 {
                            0
                        } else {
                            self.seq[j - 1]
                        };
                        let right = if i == self.cut { 0 } else { self.seq[i + 1] };
                        self.p.duplex_init
                            + self.p.exterior(rev, left, right)
                            + suffix[i + 1]
                            + second_prefix[j - 1]
                    } else {
                        self.p.hairpin(pair, &self.seq[i..=j])
                    };
                    if i != self.cut && j != self.cut + 1 && prev_split[j - 1] != INF {
                        let close = self.p.multibranch(rev, self.seq[j - 1], self.seq[i + 1]);
                        closed = closed.min(prev_split[j - 1] + self.p.ml_closing + close);
                    }
                    closed = self.internal(i, j, pair, closed);
                    if closed < INF {
                        new_stems.push((
                            j,
                            Stem {
                                energy: closed,
                                left: i as u16,
                                pair,
                            },
                        ));
                        if j <= self.cut && self.cut <= n {
                            first_suffix = first_suffix
                                .min(closed + self.exterior(i, j, pair) + suffix[j + 1]);
                        }
                    }
                }

                let col = &mut branches[offsets[j]..=offsets[j] + j];
                // In 2.8.0 the split loop's upper bound j-2 is exclusive.
                // A split directly between strands would disconnect the loop.
                let mut split = INF;
                for k in i + 1..j.saturating_sub(2) {
                    if k != self.cut && row[k] != INF && col[k + 1] != INF {
                        split = split.min(row[k] + col[k + 1]);
                    }
                }
                curr_split[j] = split;
                let mut branch = split;
                if i != self.cut + 1 && j != self.cut {
                    if closed < INF {
                        branch = branch.min(
                            closed + self.p.multibranch(pair, self.seq[i - 1], self.seq[j + 1]),
                        );
                    }
                    if i != self.cut && col[i + 1] != INF {
                        branch = branch.min(col[i + 1] + self.p.ml_base);
                    }
                    if j != self.cut + 1 && row[j - 1] != INF {
                        branch = branch.min(row[j - 1] + self.p.ml_base);
                    }
                }
                col[i] = branch;
                row[j] = branch;
            }
            if i <= self.cut && self.cut <= n {
                suffix[i] = first_suffix;
            }
            for (j, stem) in new_stems.drain(..) {
                self.stems[j].push(stem);
            }
            std::mem::swap(&mut prev_split, &mut curr_split);
        }
        self.prefix(1, n)
    }
}

pub(crate) fn monomer(seq: &[u8], p: &Params) -> f32 {
    Fold::new(seq, &[], p).fill()[seq.len()] as f32 / 100.0
}

pub(crate) fn dimer(a: &[u8], b: &[u8], second: f32, p: &Params) -> f32 {
    if a.len() + b.len() == 2 {
        return 0.0; // ViennaRNA's two-singleton fast path
    }
    let prefix = Fold::new(a, b, p).fill();
    let separate = prefix[a.len()] as f32 / 100.0 + second;
    let connected = prefix[a.len() + b.len()] as f32 / 100.0;
    connected.min(separate)
}
