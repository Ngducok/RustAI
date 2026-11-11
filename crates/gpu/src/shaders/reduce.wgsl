// Sum over the dimensions that a target shape collapses; one thread per output element.
// Words: 0 out_n, 1 kept rank, 2 reduced rank, 3 reduced total,
//        8.. kept sizes, 16.. kept source strides, 24.. reduced sizes, 32.. reduced source strides.
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let o = gid.x;
    if (o >= word(0u)) {
        return;
    }
    let kept_rank = word(1u);
    let reduced_rank = word(2u);
    var rem = o;
    var base = 0u;
    for (var d = 0u; d < 8u; d = d + 1u) {
        if (d >= kept_rank) {
            break;
        }
        let k = kept_rank - 1u - d;
        let size = word(8u + k);
        base = base + (rem % size) * word(16u + k);
        rem = rem / size;
    }
    var sum = 0.0;
    for (var r = 0u; r < word(3u); r = r + 1u) {
        var rr = r;
        var offset = base;
        for (var d = 0u; d < 8u; d = d + 1u) {
            if (d >= reduced_rank) {
                break;
            }
            let k = reduced_rank - 1u - d;
            let size = word(24u + k);
            offset = offset + (rr % size) * word(32u + k);
            rr = rr / size;
        }
        sum = sum + src[offset];
    }
    out[o] = sum;
}
