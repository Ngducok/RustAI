// Element-wise unary ops. Words: 0 n, 1 op, 2 f0, 3 f1.
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= word(0u)) {
        return;
    }
    let x = a[i];
    var r = x;
    switch word(1u) {
        case 0u: { r = max(x, 0.0); }
        case 1u: { r = x * fword(2u); }
        case 2u: { r = exp(x); }
        case 3u: { r = select(select(x, fword(3u), x > fword(3u)), fword(2u), x < fword(2u)); }
        default: { r = fword(2u); }
    }
    out[i] = r;
}
