/// 16 point ring offsets. Order is clockwise starting at top.
/// This is the FAST-16 pattern scaled to r=5 and rounded,
/// matching the paper’s "radius 5, 16 samples" design. :contentReference[oaicite:4]{index=4}
pub const RING5: [(i32,i32); 16] = [
    ( 0,-5), ( 2,-5), ( 3,-3), ( 5,-2),
    ( 5, 0), ( 5, 2), ( 3, 3), ( 2, 5),
    ( 0, 5), (-2, 5), (-3, 3), (-5, 2),
    (-5, 0), (-5,-2), (-3,-3), (-2,-5),
];

/// Optional heavier-blur ring (same angles, r=10) :contentReference[oaicite:5]{index=5}
pub const RING10: [(i32,i32); 16] = [
    ( 0,-10), ( 4,-10), ( 6, -6), (10, -4),
    (10,  0), (10,  4), ( 6,  6), ( 4, 10),
    ( 0, 10), (-4, 10), (-6,  6), (-10, 4),
    (-10, 0), (-10,-4), (-6,-6), (-4,-10),
];

#[inline]
pub fn ring_offsets(radius: u32) -> &'static [(i32,i32);16] {
    match radius {
        10 => &RING10,
        _  => &RING5,
    }
}
