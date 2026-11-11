// Gradient of an element-wise minimum. Words: 0 n, 1 first (1) or second (0) operand.
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> g: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= word(0u)) {
        return;
    }
    var mine = a[i];
    var other = b[i];
    if (word(1u) == 0u) {
        mine = b[i];
        other = a[i];
    }
    out[i] = select(select(0.0, 0.5 * g[i], mine == other), g[i], mine < other);
}
