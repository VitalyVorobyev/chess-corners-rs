//! Canonical 16-sample rings used by the ChESS detector.
/// 16 point ring offsets. Order is clockwise starting at top.
/// This is the FAST-16 pattern scaled to r=5 and rounded,
/// matching the paper’s "radius 5, 16 samples" design.
pub(crate) const RING5: [(i32, i32); 16] = [
    (0, -5),
    (2, -5),
    (3, -3),
    (5, -2),
    (5, 0),
    (5, 2),
    (3, 3),
    (2, 5),
    (0, 5),
    (-2, 5),
    (-3, 3),
    (-5, 2),
    (-5, 0),
    (-5, -2),
    (-3, -3),
    (-2, -5),
];

/// Optional heavier-blur ring (same angles, r=10)
pub(crate) const RING10: [(i32, i32); 16] = [
    (0, -10),
    (4, -10),
    (6, -6),
    (10, -4),
    (10, 0),
    (10, 4),
    (6, 6),
    (4, 10),
    (0, 10),
    (-4, 10),
    (-6, 6),
    (-10, 4),
    (-10, 0),
    (-10, -4),
    (-6, -6),
    (-4, -10),
];

/// Valid ring radii and their canonical offset tables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
#[non_exhaustive]
pub(crate) enum RingOffsets {
    /// FAST-16 offsets scaled to r=5.
    R5 = 5,
    /// Optional heavier-blur ring with r=10.
    R10 = 10,
}

impl RingOffsets {
    #[inline]
    pub const fn radius(self) -> u32 {
        self as u32
    }

    #[inline]
    pub const fn offsets(self) -> &'static [(i32, i32); 16] {
        match self {
            RingOffsets::R5 => &RING5,
            RingOffsets::R10 => &RING10,
        }
    }

    #[inline]
    pub const fn from_radius(radius: u32) -> Self {
        match radius {
            10 => RingOffsets::R10,
            _ => RingOffsets::R5,
        }
    }
}

#[inline]
/// Get the 16-sample ring offsets for the requested radius.
pub(crate) const fn ring_offsets(radius: u32) -> &'static [(i32, i32); 16] {
    RingOffsets::from_radius(radius).offsets()
}

#[cfg(test)]
mod tests {
    use super::{ring_offsets, RING10, RING5};

    #[test]
    fn ring_offsets_switch_with_radius() {
        assert_eq!(ring_offsets(5), &RING5);
        assert_eq!(ring_offsets(10), &RING10);
        // Any unknown radius currently falls back to the canonical r=5 offsets.
        assert_eq!(ring_offsets(3), &RING5);
    }
}
