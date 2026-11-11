// Every kernel receives its configuration as a block of 64 u32 words in binding 0.
struct Params {
    v: array<vec4<u32>, 16>,
}

@group(0) @binding(0) var<uniform> p: Params;

fn word(i: u32) -> u32 {
    return p.v[i / 4u][i % 4u];
}

fn fword(i: u32) -> f32 {
    return bitcast<f32>(word(i));
}
