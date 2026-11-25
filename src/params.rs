//! Nearest-neighbor energies in units of 0.01 kcal/mol. The two bundled
//! parameter files are from ViennaRNA 2.8.0 (see NOTICE-ViennaRNA.txt).

use std::collections::HashMap;
use std::str::FromStr;
use std::sync::{Arc, Mutex, OnceLock};

use crate::MatrixError;

const INF: i32 = 10_000_000;
const KELVIN_OFFSET: f64 = 273.15;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    Turner2004,
    Mathews2004,
}

impl FromStr for Model {
    type Err = MatrixError;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            "turner2004" => Ok(Self::Turner2004),
            "mathews2004" => Ok(Self::Mathews2004),
            _ => Err(MatrixError::InvalidInput(
                "Expected model='turner2004' or 'mathews2004'",
            )),
        }
    }
}

type ParamCache = Mutex<Option<(u64, Arc<Params>)>>;
static LAST_PARAMS: [ParamCache; 2] = [const { Mutex::new(None) }; 2];

/// Rows in the parameter files begin with pair CG (code 1). Mismatch
/// columns include the unknown-base entry (code 0).
type Mismatch = [[i32; 5]; 5];

struct RawParams {
    tables: HashMap<&'static str, Vec<i32>>,
    motifs: Vec<(&'static str, i32, i32)>,
}

impl RawParams {
    fn read(file: &'static str) -> Self {
        let mut tables = HashMap::new();
        let mut motifs = Vec::new();
        let mut section = "";
        for line in file.lines() {
            let line = line.split("/*").next().unwrap().trim();
            if line.is_empty() || line.starts_with("##") {
                continue;
            }
            if line == "#END" {
                break;
            }
            if let Some(name) = line.strip_prefix("# ") {
                section = name;
                continue;
            }
            let mut tokens = line.split_ascii_whitespace();
            if matches!(section, "Hexaloops" | "Tetraloops" | "Triloops") {
                let seq = tokens.next().unwrap();
                let g = tokens.next().unwrap().parse().unwrap();
                let h = tokens.next().unwrap().parse().unwrap();
                // The public API takes literal A/C/G/T. ViennaRNA recognizes
                // T for pairing but examines the original text for these motifs.
                if !seq.contains('U') {
                    motifs.push((seq, g, h));
                }
            } else {
                let row: &mut Vec<i32> = tables.entry(section).or_default();
                row.extend(tokens.map(|v| if v == "INF" { INF } else { v.parse().unwrap() }));
            }
        }
        Self { tables, motifs }
    }
}

pub(crate) struct Params {
    stack: Box<[[i32; 7]]>,
    hairpin: Vec<i32>,
    bulge: Vec<i32>,
    internal_loop: Vec<i32>,
    mismatch_h: Vec<Mismatch>,
    mismatch_i: Vec<Mismatch>,
    mismatch_1n: Vec<Mismatch>,
    mismatch_23: Vec<Mismatch>,
    exterior_stem: Vec<Mismatch>,
    ml_stem: Vec<Mismatch>,
    int11: Box<[[i32; 25]]>,
    int21: Box<[[i32; 125]]>,
    int22: Box<[[i32; 256]]>,
    motifs: HashMap<Vec<u8>, i32>,
    terminal_au: i32,
    ninio: i32,
    max_ninio: i32,
    lxc: f64,
    pub(crate) ml_closing: i32,
    pub(crate) ml_base: i32,
    pub(crate) duplex_init: i32,
}

fn rows<const N: usize>(values: &[i32]) -> Box<[[i32; N]]> {
    let (rows, remainder) = values.as_chunks::<N>();
    assert!(remainder.is_empty());
    rows.into()
}

fn mismatches(values: &[i32]) -> Vec<Mismatch> {
    let (tables, remainder) = values.as_chunks::<25>();
    assert!(remainder.is_empty());
    tables
        .iter()
        .map(|row| std::array::from_fn(|i| std::array::from_fn(|j| row[i * 5 + j])))
        .collect()
}

impl Params {
    pub(crate) fn shared(model: Model, temperature: f64) -> Result<Arc<Self>, MatrixError> {
        if !temperature.is_finite() || temperature <= -KELVIN_OFFSET {
            return Err(MatrixError::InvalidInput(
                "Temperature must be finite and above absolute zero (Celsius)",
            ));
        }
        let dna = model == Model::Mathews2004;
        let mut cached = LAST_PARAMS[dna as usize]
            .lock()
            .map_err(|_| MatrixError::Runtime("Parameter cache poisoned".into()))?;
        match cached.as_ref() {
            Some((key, params)) if *key == temperature.to_bits() => Ok(params.clone()),
            _ => {
                let params = Arc::new(Self::new(temperature, dna));
                *cached = Some((temperature.to_bits(), params.clone()));
                Ok(params)
            }
        }
    }
    pub(crate) fn new(temperature: f64, dna: bool) -> Self {
        static RNA: OnceLock<RawParams> = OnceLock::new();
        static DNA: OnceLock<RawParams> = OnceLock::new();
        let raw = if dna {
            DNA.get_or_init(|| RawParams::read(include_str!("../data/dna_mathews2004.par")))
        } else {
            RNA.get_or_init(|| RawParams::read(include_str!("../data/rna_turner2004.par")))
        };
        let factor = (temperature + KELVIN_OFFSET) / (37.0 + KELVIN_OFFSET);
        let scale = |g: i32, h: i32| (f64::from(h) - f64::from(h - g) * factor) as i32;
        let table = |name: &str| -> Vec<i32> {
            let enthalpies = &raw.tables[format!("{name}_enthalpies").as_str()];
            raw.tables[name]
                .iter()
                .zip(enthalpies)
                .map(|(&g, &h)| scale(g, h))
                .collect()
        };
        let scalar = |name: &str, index: usize| {
            let values = &raw.tables[name];
            scale(values[index], values[index + 1])
        };
        let terminal_au = scalar("Misc", 2);
        let dangle5 = rows::<5>(&table("dangle5"));
        let dangle3 = rows::<5>(&table("dangle3"));
        // Precompute all four dangle configurations. In these two tables
        // alone, index zero means the flank lies beyond the strand end.
        let stems = |name: &str, branch_cost: i32| {
            let mut entries = mismatches(&table(name));
            for (p, entry) in entries.iter_mut().enumerate() {
                let end_cost = if p + 1 > 2 { terminal_au } else { 0 };
                for (i, row) in entry.iter_mut().enumerate() {
                    for (j, value) in row.iter_mut().enumerate() {
                        let dangle = match (i, j) {
                            (0, 0) => 0,
                            (0, _) => dangle3[p][j],
                            (_, 0) => dangle5[p][i],
                            _ => *value,
                        };
                        *value = branch_cost + end_cost + dangle.min(0);
                    }
                }
            }
            entries
        };

        let motifs = raw
            .motifs
            .iter()
            .map(|&(seq, g, h)| {
                let bases = seq
                    .bytes()
                    .map(|base| match base {
                        b'A' => 1,
                        b'C' => 2,
                        b'G' => 3,
                        b'T' => 4,
                        _ => unreachable!(),
                    })
                    .collect();
                (bases, scale(g, h))
            })
            .collect();

        Self {
            stack: rows(&table("stack")),
            hairpin: table("hairpin"),
            bulge: table("bulge"),
            internal_loop: table("internal"),
            mismatch_h: mismatches(&table("mismatch_hairpin")),
            mismatch_i: mismatches(&table("mismatch_internal")),
            mismatch_1n: mismatches(&table("mismatch_internal_1n")),
            mismatch_23: mismatches(&table("mismatch_internal_23")),
            exterior_stem: stems("mismatch_exterior", 0),
            ml_stem: stems("mismatch_multi", scalar("ML_params", 4)),
            int11: rows(&table("int11")),
            int21: rows(&table("int21")),
            int22: rows(&table("int22")),
            motifs,
            terminal_au,
            ninio: scalar("NINIO", 0),
            max_ninio: raw.tables["NINIO"][2],
            lxc: 107.856 * factor,
            ml_closing: scalar("ML_params", 2),
            ml_base: scalar("ML_params", 0),
            duplex_init: scalar("Misc", 0),
        }
    }

    #[inline]
    pub(crate) fn stack(&self, outer: u8, inner_rev: u8) -> i32 {
        self.stack[(outer - 1) as usize][(inner_rev - 1) as usize]
    }

    #[inline]
    pub(crate) fn exterior(&self, pair: u8, flank5: u8, flank3: u8) -> i32 {
        self.exterior_stem[(pair - 1) as usize][flank5 as usize][flank3 as usize]
    }

    #[inline]
    pub(crate) fn multibranch(&self, pair: u8, flank5: u8, flank3: u8) -> i32 {
        self.ml_stem[(pair - 1) as usize][flank5 as usize][flank3 as usize]
    }

    fn loop_cost(&self, costs: &[i32], size: usize) -> i32 {
        if size <= 30 {
            costs[size]
        } else {
            costs[30] + (self.lxc * (size as f64 / 30.).ln()) as i32
        }
    }

    pub(crate) fn hairpin(&self, pair: u8, sequence: &[u8]) -> i32 {
        let size = sequence.len() - 2;
        let cost = self.loop_cost(&self.hairpin, size);
        if size < 3 {
            return cost;
        }
        if matches!(size, 3 | 4 | 6) {
            if let Some(&energy) = self.motifs.get(sequence) {
                return energy;
            }
            if size == 3 {
                return cost + self.end_penalty(pair);
            }
        }
        cost + self.mismatch_h[(pair - 1) as usize][sequence[1] as usize][sequence[size] as usize]
    }

    #[inline]
    fn end_penalty(&self, pair: u8) -> i32 {
        if pair > 2 { self.terminal_au } else { 0 }
    }

    /// Bases around the nested pair (k,l) are supplied in the ViennaRNA
    /// order i+1, j-1, k-1, l+1. The nested pair type is read as (l,k).
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn internal(
        &self,
        n1: usize,
        n2: usize,
        outer: u8,
        inner_rev: u8,
        si1: u8,
        sj1: u8,
        sp1: u8,
        sq1: u8,
    ) -> i32 {
        let small = n1.min(n2);
        let large = n1.max(n2);
        let p = usize::from(outer - 1);
        let q = usize::from(inner_rev - 1);
        let si = usize::from(si1);
        let sj = usize::from(sj1);
        let sp = usize::from(sp1);
        let sq = usize::from(sq1);
        match (small, large) {
            (0, 0) => return self.stack(outer, inner_rev),
            (0, 1) => return self.bulge[1] + self.stack(outer, inner_rev),
            (0, _) => {
                return self.loop_cost(&self.bulge, large)
                    + self.end_penalty(outer)
                    + self.end_penalty(inner_rev);
            }
            (1, 1) => return self.int11[p * 7 + q][si * 5 + sj],
            (1, 2) if n1 == 1 => return self.int21[p * 7 + q][(si * 5 + sq) * 5 + sj],
            (1, 2) => return self.int21[q * 7 + p][(sq * 5 + si) * 5 + sp],
            (2, 2) => {
                // int22 alone omits unknown bases and the nonstandard pair.
                let column = [si, sp, sq, sj]
                    .into_iter()
                    .fold(0, |idx, b| idx * 4 + (b - 1));
                return self.int22[p * 6 + q][column];
            }
            _ => {}
        }
        let (mismatch, asymmetry) = match (small, large) {
            (2, 3) => (&self.mismatch_23, self.ninio),
            _ => {
                let asymmetry = (self.ninio * (large - small) as i32).min(self.max_ninio);
                let mismatch = if small == 1 {
                    &self.mismatch_1n
                } else {
                    &self.mismatch_i
                };
                (mismatch, asymmetry)
            }
        };
        self.loop_cost(&self.internal_loop, small + large)
            + asymmetry
            + mismatch[p][si][sj]
            + mismatch[q][sq][sp]
    }
}
