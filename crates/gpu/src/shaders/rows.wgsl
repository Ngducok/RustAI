// Row-wise kernels over a rows x cols matrix. Words: 0 rows, 1 cols, 2 mode.
//   mode 0: cross-entropy loss (one workgroup, writes the mean to out[0])
//   mode 1: cross-entropy gradient scaled by upstream[0] / rows
//   mode 2: log-softmax
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read> labels: array<f32>;
@group(0) @binding(3) var<storage, read> upstream: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

var<workgroup> partial: array<f32, 64>;

fn row_max(base: u32, cols: u32) -> f32 {
    var mx = logits[base];
    for (var c = 1u; c < cols; c = c + 1u) {
        mx = max(mx, logits[base + c]);
    }
    return mx;
}

fn row_sum(base: u32, cols: u32, mx: f32) -> f32 {
    var s = 0.0;
    for (var c = 0u; c < cols; c = c + 1u) {
        s = s + exp(logits[base + c] - mx);
    }
    return s;
}

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let rows = word(0u);
    let cols = word(1u);
    let mode = word(2u);
    if (mode == 0u) {
        var acc = 0.0;
        for (var r = lid.x; r < rows; r = r + 64u) {
            let base = r * cols;
            let mx = row_max(base, cols);
            let s = row_sum(base, cols, mx);
            acc = acc - ((logits[base + u32(labels[r])] - mx) - log(s));
        }
        partial[lid.x] = acc;
        workgroupBarrier();
        if (lid.x == 0u) {
            var total = 0.0;
            for (var i = 0u; i < 64u; i = i + 1u) {
                total = total + partial[i];
            }
            out[0] = total / f32(max(rows, 1u));
        }
        return;
    }
    let r = wid.x * 64u + lid.x;
    if (r >= rows) {
        return;
    }
    let base = r * cols;
    let mx = row_max(base, cols);
    let s = row_sum(base, cols, mx);
    if (mode == 1u) {
        let scale = upstream[0] / f32(max(rows, 1u));
        let label = u32(labels[r]);
        for (var c = 0u; c < cols; c = c + 1u) {
            var g = exp(logits[base + c] - mx) / s * scale;
            if (c == label) {
                g = g - scale;
            }
            out[base + c] = g;
        }
    } else {
        let log_sum = log(s);
        for (var c = 0u; c < cols; c = c + 1u) {
            out[base + c] = (logits[base + c] - mx) - log_sum;
        }
    }
}
