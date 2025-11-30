//! Canonical 16-sample rings used by the ChESS detector.
/// 16 point ring offsets. Order is clockwise starting at top.
/// This is the FAST-16 pattern scaled to r=5 and rounded,
/// matching the paper’s "radius 5, 16 samples" design.
pub const RING5: [(i32, i32); 16] = [
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
pub const RING10: [(i32, i32); 16] = [
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
pub enum RingOffsets {
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
pub const fn ring_offsets(radius: u32) -> &'static [(i32, i32); 16] {
    RingOffsets::from_radius(radius).offsets()
}
