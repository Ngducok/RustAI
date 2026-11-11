// Element-wise binary ops with broadcasting (also used for copies and transposes).
// Words: 0 n, 1 rank, 2 op, 3 f0, 4 f1, 8.. out shape, 16.. a strides, 24.. b strides.
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= word(0u)) {
        return;
    }
    let rank = word(1u);
    var rem = i;
    var oa = 0u;
    var ob = 0u;
    for (var d = 0u; d < 8u; d = d + 1u) {
        if (d >= rank) {
            break;
        }
        let k = rank - 1u - d;
        let size = word(8u + k);
        let coord = rem % size;
        rem = rem / size;
        oa = oa + coord * word(16u + k);
        ob = ob + coord * word(24u + k);
    }
    let x = a[oa];
    let y = b[ob];
    var r = x;
    switch word(2u) {
        case 0u: { r = x + y; }
        case 1u: { r = x - y; }
        case 2u: { r = x * y; }
        case 3u: { r = min(x, y); }
        case 4u: { r = select(0.0, y, x > 0.0); }
        case 5u: { r = select(0.0, y, x >= fword(3u) && x <= fword(4u)); }
        default: { r = x; }
    }
    out[i] = r;
}
