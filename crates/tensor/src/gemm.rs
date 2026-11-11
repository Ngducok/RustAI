//! Blocked single-precision matrix multiply.
//!
//! Both operands are packed into contiguous tiles (B by column tile, A by row tile), K is split
//! into cache-sized blocks, and a register-tiled micro-kernel keeps its accumulators in registers.
//! Either operand may be stored transposed; packing absorbs the difference. Every output element
//! is accumulated in ascending-k order, so results depend on neither blocking, thread count nor
//! operand layout.

use std::cell::RefCell;
use std::thread::LocalKey;

use rayon::prelude::*;

const KC: usize = 256;
const ROW_STEP: usize = 12;
const MAX_CHUNK_ROWS: usize = 96;
const PARALLEL_MIN_MACS: usize = 1 << 20;

/// How an operand is stored relative to the logical matrix it represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    RowMajor,
    Transposed,
}

/// Work applied to each output element once its dot product is complete, in this order:
/// add `bias[column]`, clamp at zero, then zero wherever `mask` is not positive.
#[derive(Clone, Copy, Default)]
pub(crate) struct Epilogue<'a> {
    pub bias: Option<&'a [f32]>,
    pub relu: bool,
    pub mask: Option<&'a [f32]>,
}

/// An epilogue positioned at the top-left element of one micro-tile.
#[derive(Clone, Copy)]
struct Finish<'a> {
    bias: Option<&'a [f32]>,
    relu: bool,
    mask: Option<&'a [f32]>,
}

impl Finish<'_> {
    #[inline(always)]
    fn apply(&self, value: f32, row: usize, ldc: usize, column: usize) -> f32 {
        let mut x = value;
        if let Some(bias) = self.bias {
            x += bias[column];
        }
        if self.relu {
            x = x.max(0.0);
        }
        match self.mask {
            Some(mask) if mask[row * ldc + column] > 0.0 => x,
            Some(_) => 0.0,
            None => x,
        }
    }
}

/// How one micro-tile reaches memory.
struct Store<'a> {
    ldc: usize,
    cols: usize,
    first: bool,
    finish: Option<Finish<'a>>,
}

/// The product of a logical `m x k` matrix `a` and `k x n` matrix `b`.
pub(crate) struct Product<'a> {
    pub a: &'a [f32],
    pub a_layout: Layout,
    pub b: &'a [f32],
    pub b_layout: Layout,
    pub m: usize,
    pub k: usize,
    pub n: usize,
    pub epilogue: Epilogue<'a>,
}

/// One K-block for a run of output rows: packed A tiles, packed B tiles, and where to write.
struct Block<'a> {
    a: &'a [f32],
    rows: usize,
    n: usize,
    panel: &'a [f32],
    first: bool,
    last: bool,
    row0: usize,
    epilogue: Epilogue<'a>,
}

type RowsKernel = unsafe fn(&Block, &mut [f32]);

thread_local! {
    static PACKED_A: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    static PACKED_B: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Whether products are accumulated with fused multiply-add (changes rounding, not the algorithm).
pub fn uses_fma() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

pub(crate) fn matmul(product: &Product, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if uses_fma() {
        return gemm::<6, 16>(rows_fma, product, out);
    }
    gemm::<4, 8>(rows_portable, product, out)
}

unsafe fn rows_portable(block: &Block, c: &mut [f32]) {
    rows::<4, 8, false>(block, c)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
fn rows_fma(block: &Block, c: &mut [f32]) {
    rows::<6, 16, true>(block, c)
}

fn with_scratch<R>(
    key: &'static LocalKey<RefCell<Vec<f32>>>,
    len: usize,
    f: impl FnOnce(&mut [f32]) -> R,
) -> R {
    key.with(|cell| match cell.try_borrow_mut() {
        Ok(mut buffer) => {
            if buffer.len() < len {
                buffer.resize(len, 0.0);
            }
            f(&mut buffer[..len])
        }
        Err(_) => f(&mut vec![0.0; len]),
    })
}

fn pack_b<const NR: usize>(p: &Product, pc: usize, kc: usize, panel: &mut [f32]) {
    let (b, n, k) = (p.b, p.n, p.k);
    for (tile_index, tile) in panel.chunks_exact_mut(kc * NR).enumerate() {
        let col0 = tile_index * NR;
        let cols = NR.min(n - col0);
        for (step, dst) in tile.as_chunks_mut::<NR>().0.iter_mut().enumerate() {
            let depth = pc + step;
            match p.b_layout {
                Layout::RowMajor => dst[..cols].copy_from_slice(&b[depth * n + col0..][..cols]),
                Layout::Transposed => {
                    for (c, slot) in dst[..cols].iter_mut().enumerate() {
                        *slot = b[(col0 + c) * k + depth];
                    }
                }
            }
            dst[cols..].fill(0.0);
        }
    }
}

/// Packs `rows` rows starting at `row0`: full tiles of `MR` rows stored depth-major, then any
/// leftover rows stored one after another as plain strips.
fn pack_a<const MR: usize>(
    p: &Product,
    row0: usize,
    rows: usize,
    pc: usize,
    kc: usize,
    out: &mut [f32],
) {
    let (a, m, k) = (p.a, p.m, p.k);
    let element = |row: usize, depth: usize| match p.a_layout {
        Layout::RowMajor => a[row * k + depth],
        Layout::Transposed => a[depth * m + row],
    };
    let full = rows / MR * MR;
    let (tiles, strips) = out.split_at_mut(full * kc);
    for (tile_index, tile) in tiles.chunks_exact_mut(MR * kc).enumerate() {
        for (step, column) in tile.as_chunks_mut::<MR>().0.iter_mut().enumerate() {
            for (r, slot) in column.iter_mut().enumerate() {
                *slot = element(row0 + tile_index * MR + r, pc + step);
            }
        }
    }
    for (r, strip) in strips.chunks_exact_mut(kc).enumerate() {
        for (step, slot) in strip.iter_mut().enumerate() {
            *slot = element(row0 + full + r, pc + step);
        }
    }
}

fn gemm<const MR: usize, const NR: usize>(kernel: RowsKernel, p: &Product, out: &mut [f32]) {
    let threads = rayon::current_num_threads();
    let parallel = threads > 1 && p.m * p.k * p.n >= PARALLEL_MIN_MACS;
    let chunk_rows = if parallel {
        p.m.div_ceil(2 * threads)
            .next_multiple_of(ROW_STEP)
            .clamp(ROW_STEP, MAX_CHUNK_ROWS)
    } else {
        MAX_CHUNK_ROWS
    };
    let tiles = p.n.div_ceil(NR);

    with_scratch(&PACKED_B, KC.min(p.k) * tiles * NR, |scratch| {
        for pc in (0..p.k).step_by(KC) {
            let kc = KC.min(p.k - pc);
            pack_b::<NR>(p, pc, kc, &mut scratch[..kc * tiles * NR]);
            let panel = &scratch[..kc * tiles * NR];
            let work = |(chunk, c_rows): (usize, &mut [f32])| {
                let rows = c_rows.len() / p.n;
                with_scratch(&PACKED_A, rows * kc, |packed| {
                    pack_a::<MR>(p, chunk * chunk_rows, rows, pc, kc, packed);
                    let block = Block {
                        a: packed,
                        rows,
                        n: p.n,
                        panel,
                        first: pc == 0,
                        last: pc + kc == p.k,
                        row0: chunk * chunk_rows,
                        epilogue: p.epilogue,
                    };
                    // SAFETY: `kernel` was selected by `matmul` after checking the CPU supports it.
                    unsafe { kernel(&block, c_rows) }
                })
            };
            if parallel {
                out.par_chunks_mut(chunk_rows * p.n)
                    .enumerate()
                    .for_each(work);
            } else {
                out.chunks_mut(chunk_rows * p.n).enumerate().for_each(work);
            }
        }
    });
}

#[inline(always)]
fn finish_at<'a>(block: &Block<'a>, row: usize, column: usize) -> Option<Finish<'a>> {
    let e = block.epilogue;
    if !block.last || (e.bias.is_none() && !e.relu && e.mask.is_none()) {
        return None;
    }
    Some(Finish {
        bias: e.bias.map(|bias| &bias[column..]),
        relu: e.relu,
        mask: e
            .mask
            .map(|mask| &mask[(block.row0 + row) * block.n + column..]),
    })
}

#[inline(always)]
fn rows<const MR: usize, const NR: usize, const FMA: bool>(block: &Block, c: &mut [f32]) {
    let kc = block.a.len() / block.rows;
    let full = block.rows / MR * MR;
    for (tile_index, b_tile) in block.panel.chunks_exact(kc * NR).enumerate() {
        let col0 = tile_index * NR;
        let cols = NR.min(block.n - col0);
        let store = |row| Store {
            ldc: block.n,
            cols,
            first: block.first,
            finish: finish_at(block, row, col0),
        };
        for row in (0..full).step_by(MR) {
            let a_tile = &block.a[row * kc..][..MR * kc];
            let c_tile = &mut c[row * block.n + col0..];
            micro::<MR, NR, FMA>(a_tile, b_tile, c_tile, &store(row));
        }
        for row in full..block.rows {
            let a_strip = &block.a[row * kc..][..kc];
            let c_tile = &mut c[row * block.n + col0..];
            micro::<1, NR, FMA>(a_strip, b_tile, c_tile, &store(row));
        }
    }
}

#[inline(always)]
fn multiply_add<const FMA: bool>(a: f32, b: f32, acc: f32) -> f32 {
    if FMA {
        a.mul_add(b, acc)
    } else {
        a * b + acc
    }
}

#[inline(always)]
fn micro<const MR: usize, const NR: usize, const FMA: bool>(
    a_tile: &[f32],
    b_tile: &[f32],
    c: &mut [f32],
    store: &Store,
) {
    let (ldc, cols) = (store.ldc, store.cols);
    let mut acc = [[0.0f32; NR]; MR];
    if !store.first {
        for (r, row) in acc.iter_mut().enumerate() {
            row[..cols].copy_from_slice(&c[r * ldc..][..cols]);
        }
    }
    for (a_col, b_row) in a_tile
        .as_chunks::<MR>()
        .0
        .iter()
        .zip(b_tile.as_chunks::<NR>().0)
    {
        for r in 0..MR {
            for j in 0..NR {
                acc[r][j] = multiply_add::<FMA>(a_col[r], b_row[j], acc[r][j]);
            }
        }
    }
    for (r, row) in acc.iter().enumerate() {
        let dst = &mut c[r * ldc..][..cols];
        match &store.finish {
            None => dst.copy_from_slice(&row[..cols]),
            Some(finish) => {
                for (j, (slot, &value)) in dst.iter_mut().zip(&row[..cols]).enumerate() {
                    *slot = finish.apply(value, r, ldc, j);
                }
            }
        }
    }
}

/// The result when `k == 0`: every dot product is zero, so only the epilogue remains.
pub(crate) fn empty_product(n: usize, epilogue: &Epilogue, out: &mut [f32]) {
    let finish = Finish {
        bias: epilogue.bias,
        relu: epilogue.relu,
        mask: epilogue.mask,
    };
    for (index, slot) in out.iter_mut().enumerate() {
        *slot = finish.apply(0.0, index / n, n, index % n);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;

    const LAYOUTS: [Layout; 2] = [Layout::RowMajor, Layout::Transposed];

    fn random(len: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        (0..len).map(|_| rng.uniform(-1.0, 1.0)).collect()
    }

    fn product<'a>(
        a: (&'a [f32], Layout),
        b: (&'a [f32], Layout),
        dims: (usize, usize, usize),
    ) -> Product<'a> {
        Product {
            a: a.0,
            a_layout: a.1,
            b: b.0,
            b_layout: b.1,
            m: dims.0,
            k: dims.1,
            n: dims.2,
            epilogue: Epilogue::default(),
        }
    }

    fn at(data: &[f32], layout: Layout, rows: usize, cols: usize, i: usize, j: usize) -> f32 {
        match layout {
            Layout::RowMajor => data[i * cols + j],
            Layout::Transposed => data[j * rows + i],
        }
    }

    fn ordered_reference(p: &Product, fused: bool) -> Vec<f32> {
        (0..p.m * p.n)
            .map(|index| {
                let (i, j) = (index / p.n, index % p.n);
                (0..p.k).fold(0.0f32, |acc, step| {
                    let x = at(p.a, p.a_layout, p.m, p.k, i, step);
                    let y = at(p.b, p.b_layout, p.k, p.n, step, j);
                    if fused {
                        x.mul_add(y, acc)
                    } else {
                        x * y + acc
                    }
                })
            })
            .collect()
    }

    fn assert_same(label: &str, actual: &[f32], expected: &[f32]) {
        for (i, (x, y)) in actual.iter().zip(expected).enumerate() {
            assert!(x == y, "{label}: element {i} is {x}, reference gives {y}");
        }
    }

    fn portable(p: &Product) -> Vec<f32> {
        let mut out = vec![f32::NAN; p.m * p.n];
        gemm::<4, 8>(rows_portable, p, &mut out);
        out
    }

    fn dispatched(p: &Product) -> Vec<f32> {
        let mut out = vec![f32::NAN; p.m * p.n];
        matmul(p, &mut out);
        out
    }

    #[test]
    fn every_layout_matches_the_ordered_reference_across_tile_boundaries() {
        for m in [1, 5, 6, 13, 97] {
            for n in [1, 9, 16, 17, 33] {
                for k in [1, 7, 256, 257] {
                    let (a, b) = (random(m * k, 1), random(k * n, 2));
                    for a_layout in LAYOUTS {
                        for b_layout in LAYOUTS {
                            let p = product((&a, a_layout), (&b, b_layout), (m, k, n));
                            let label = format!("{m}x{k}x{n} {a_layout:?}/{b_layout:?}");
                            assert_same(
                                &format!("portable {label}"),
                                &portable(&p),
                                &ordered_reference(&p, false),
                            );
                            assert_same(
                                &format!("dispatched {label}"),
                                &dispatched(&p),
                                &ordered_reference(&p, uses_fma()),
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn transposed_operands_equal_explicit_transposes_exactly() {
        let (m, k, n) = (37, 70, 29);
        let a = random(m * k, 11);
        let b = random(k * n, 12);
        let transpose = |data: &[f32], rows: usize, cols: usize| -> Vec<f32> {
            (0..rows * cols)
                .map(|i| data[(i % rows) * cols + i / rows])
                .collect()
        };
        let (a_t, b_t) = (transpose(&a, m, k), transpose(&b, k, n));
        let plain = dispatched(&product(
            (&a, Layout::RowMajor),
            (&b, Layout::RowMajor),
            (m, k, n),
        ));
        assert_same(
            "A stored transposed",
            &dispatched(&product(
                (&a_t, Layout::Transposed),
                (&b, Layout::RowMajor),
                (m, k, n),
            )),
            &plain,
        );
        assert_same(
            "B stored transposed",
            &dispatched(&product(
                (&a, Layout::RowMajor),
                (&b_t, Layout::Transposed),
                (m, k, n),
            )),
            &plain,
        );
    }

    #[test]
    fn parallel_and_sequential_paths_agree_exactly() {
        let (m, k, n) = (130, 300, 100);
        assert!(m * k * n >= PARALLEL_MIN_MACS);
        let (a, b) = (random(m * k, 3), random(k * n, 4));
        for a_layout in LAYOUTS {
            let p = product((&a, a_layout), (&b, Layout::RowMajor), (m, k, n));
            let expected = ordered_reference(&p, uses_fma());
            assert_same("default pool", &dispatched(&p), &expected);
            let single = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap();
            assert_same("one thread", &single.install(|| dispatched(&p)), &expected);
            let many = rayon::ThreadPoolBuilder::new()
                .num_threads(5)
                .build()
                .unwrap();
            assert_same("five threads", &many.install(|| dispatched(&p)), &expected);
        }
    }

    #[test]
    fn every_output_element_is_overwritten() {
        let (a, b) = (random(7 * 5, 5), random(5 * 9, 6));
        let out = dispatched(&product(
            (&a, Layout::RowMajor),
            (&b, Layout::RowMajor),
            (7, 5, 9),
        ));
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn reentrant_calls_fall_back_to_private_scratch() {
        let (a, b) = (random(6 * 8, 7), random(8 * 6, 8));
        let p = product((&a, Layout::RowMajor), (&b, Layout::RowMajor), (6, 8, 6));
        let expected = ordered_reference(&p, uses_fma());
        let nested = with_scratch(&PACKED_B, 16, |_| {
            with_scratch(&PACKED_A, 16, |_| dispatched(&p))
        });
        assert_same("nested", &nested, &expected);
    }

    #[test]
    fn repeated_calls_are_deterministic() {
        let (m, k, n) = (64, 64, 64);
        let (a, b) = (random(m * k, 9), random(k * n, 10));
        let p = product(
            (&a, Layout::Transposed),
            (&b, Layout::Transposed),
            (m, k, n),
        );
        let first = dispatched(&p);
        assert_same("repeat", &dispatched(&p), &first);
    }

    fn apply_epilogue(raw: &[f32], n: usize, e: &Epilogue) -> Vec<f32> {
        raw.iter()
            .enumerate()
            .map(|(index, &value)| {
                let mut x = value;
                if let Some(bias) = e.bias {
                    x += bias[index % n];
                }
                if e.relu {
                    x = x.max(0.0);
                }
                match e.mask {
                    Some(mask) if mask[index] > 0.0 => x,
                    Some(_) => 0.0,
                    None => x,
                }
            })
            .collect()
    }

    fn epilogues<'a>(bias: &'a [f32], mask: &'a [f32]) -> Vec<Epilogue<'a>> {
        let (bias, mask) = (Some(bias), Some(mask));
        vec![
            Epilogue {
                bias,
                relu: false,
                mask: None,
            },
            Epilogue {
                bias,
                relu: true,
                mask: None,
            },
            Epilogue {
                bias: None,
                relu: true,
                mask: None,
            },
            Epilogue {
                bias: None,
                relu: false,
                mask,
            },
            Epilogue {
                bias,
                relu: true,
                mask,
            },
        ]
    }

    #[test]
    fn epilogues_match_separate_elementwise_passes_exactly() {
        for m in [1, 6, 13, 97] {
            for n in [1, 9, 17, 33] {
                for k in [1, 7, 257] {
                    let (a, b) = (random(m * k, 21), random(k * n, 22));
                    let (bias, mask) = (random(n, 23), random(m * n, 24));
                    for e in epilogues(&bias, &mask) {
                        let mut p =
                            product((&a, Layout::RowMajor), (&b, Layout::RowMajor), (m, k, n));
                        p.epilogue = e;
                        let label = format!(
                            "{m}x{k}x{n} bias={} relu={} mask={}",
                            e.bias.is_some(),
                            e.relu,
                            e.mask.is_some()
                        );
                        let plain =
                            product((&a, Layout::RowMajor), (&b, Layout::RowMajor), (m, k, n));
                        assert_same(
                            &format!("portable {label}"),
                            &portable(&p),
                            &apply_epilogue(&portable(&plain), n, &e),
                        );
                        assert_same(
                            &format!("dispatched {label}"),
                            &dispatched(&p),
                            &apply_epilogue(&dispatched(&plain), n, &e),
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn epilogues_hold_on_the_parallel_path_and_with_transposed_operands() {
        let (m, k, n) = (130, 300, 100);
        let (a, b) = (random(m * k, 25), random(k * n, 26));
        let (bias, mask) = (random(n, 27), random(m * n, 28));
        let e = Epilogue {
            bias: Some(&bias),
            relu: true,
            mask: Some(&mask),
        };
        for a_layout in LAYOUTS {
            let mut p = product((&a, a_layout), (&b, Layout::RowMajor), (m, k, n));
            let expected = apply_epilogue(&ordered_reference(&p, uses_fma()), n, &e);
            p.epilogue = e;
            assert_same("default pool", &dispatched(&p), &expected);
            let single = rayon::ThreadPoolBuilder::new()
                .num_threads(1)
                .build()
                .unwrap();
            assert_same("one thread", &single.install(|| dispatched(&p)), &expected);
            let many = rayon::ThreadPoolBuilder::new()
                .num_threads(5)
                .build()
                .unwrap();
            assert_same("five threads", &many.install(|| dispatched(&p)), &expected);
        }
    }
}
