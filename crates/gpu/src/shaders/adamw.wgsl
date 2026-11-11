// One AdamW step over a parameter tensor, entirely on the GPU.
// Words: 0 n, 1 lr, 2 beta1, 3 beta2, 4 eps, 5 weight decay, 6 bias correction 1, 7 bias correction 2.
@group(0) @binding(1) var<storage, read_write> param: array<f32>;
@group(0) @binding(2) var<storage, read> grad: array<f32>;
@group(0) @binding(3) var<storage, read_write> first: array<f32>;
@group(0) @binding(4) var<storage, read_write> second: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= word(0u)) {
        return;
    }
    let g = grad[i];
    let beta1 = fword(2u);
    let beta2 = fword(3u);
    let m = beta1 * first[i] + (1.0 - beta1) * g;
    let v = beta2 * second[i] + (1.0 - beta2) * g * g;
    first[i] = m;
    second[i] = v;
    let step = (m / fword(6u)) / (sqrt(v / fword(7u)) + fword(4u));
    param[i] = param[i] - fword(1u) * (step + fword(5u) * param[i]);
}
