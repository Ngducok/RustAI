// Tiled matrix multiply with strided operands and an optional epilogue.
// Each workgroup computes a 64x64 tile; each of its 16x16 threads computes 4x4 outputs.
// Words: 0 m, 1 n, 2 k, 3 a stride(i), 4 a stride(p), 5 b stride(p), 6 b stride(j),
//        7 flags (1 bias, 2 relu, 4 mask).
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> mask: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

var<workgroup> tile_a: array<f32, 1024>;
var<workgroup> tile_b: array<f32, 1024>;

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let m = word(0u);
    let n = word(1u);
    let k = word(2u);
    let a_i = word(3u);
    let a_p = word(4u);
    let b_p = word(5u);
    let b_j = word(6u);
    let flags = word(7u);
    let tid = lid.y * 16u + lid.x;
    let row0 = wid.y * 64u;
    let col0 = wid.x * 64u;

    var acc: array<f32, 16>;
    for (var p0 = 0u; p0 < k; p0 = p0 + 16u) {
        for (var s = 0u; s < 4u; s = s + 1u) {
            let idx = tid + s * 256u;
            let ra = idx / 16u;
            let pa = p0 + idx % 16u;
            let gi = row0 + ra;
            tile_a[idx] = select(0.0, a[gi * a_i + pa * a_p], gi < m && pa < k);
            let pb = p0 + idx / 64u;
            let gj = col0 + idx % 64u;
            tile_b[idx] = select(0.0, b[pb * b_p + gj * b_j], pb < k && gj < n);
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            for (var ri = 0u; ri < 4u; ri = ri + 1u) {
                let av = tile_a[(lid.y * 4u + ri) * 16u + kk];
                for (var ci = 0u; ci < 4u; ci = ci + 1u) {
                    acc[ri * 4u + ci] = acc[ri * 4u + ci] + av * tile_b[kk * 64u + lid.x * 4u + ci];
                }
            }
        }
        workgroupBarrier();
    }

    for (var ri = 0u; ri < 4u; ri = ri + 1u) {
        for (var ci = 0u; ci < 4u; ci = ci + 1u) {
            let i = row0 + lid.y * 4u + ri;
            let j = col0 + lid.x * 4u + ci;
            if (i < m && j < n) {
                var x = acc[ri * 4u + ci];
                if ((flags & 1u) != 0u) {
                    x = x + bias[j];
                }
                if ((flags & 2u) != 0u) {
                    x = max(x, 0.0);
                }
                if ((flags & 4u) != 0u) {
                    x = select(0.0, x, mask[i * n + j] > 0.0);
                }
                out[i * n + j] = x;
            }
        }
    }
}
