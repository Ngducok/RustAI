#[derive(Debug, Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }

    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_gives_same_sequence() {
        let (mut a, mut b) = (Rng::new(9), Rng::new(9));
        assert!((0..100).all(|_| a.next_u64() == b.next_u64()));
        assert_ne!(Rng::new(9).next_u64(), Rng::new(10).next_u64());
    }

    #[test]
    fn zero_seed_does_not_get_stuck() {
        let mut rng = Rng::new(0);
        assert_ne!(rng.next_u64(), rng.next_u64());
    }

    #[test]
    fn floats_are_in_range_and_unbiased() {
        let mut rng = Rng::new(1);
        let samples: Vec<f32> = (0..20_000).map(|_| rng.next_f32()).collect();
        assert!(samples.iter().all(|&x| (0.0..1.0).contains(&x)));
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!((mean - 0.5).abs() < 0.01, "mean {mean}");
        let below_quarter = samples.iter().filter(|&&x| x < 0.25).count() as f32 / 20_000.0;
        assert!(
            (below_quarter - 0.25).abs() < 0.015,
            "fraction {below_quarter}"
        );
    }

    #[test]
    fn uniform_respects_bounds() {
        let mut rng = Rng::new(2);
        assert!((0..1000).all(|_| (-3.0..5.0).contains(&rng.uniform(-3.0, 5.0))));
    }

    #[test]
    fn below_covers_every_value() {
        let mut rng = Rng::new(3);
        let mut seen = [false; 7];
        for _ in 0..500 {
            seen[rng.below(7)] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }
}
