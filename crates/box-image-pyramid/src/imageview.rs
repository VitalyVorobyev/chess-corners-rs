/// A borrowed, non-owning view into a row-major u8 grayscale image.
///
/// `ImageView` does not allocate or copy; it is a lightweight wrapper
/// around an existing byte slice. Use it to pass image data into
/// `build_pyramid` without transferring ownership.
///
/// Invariant: `data.len() == width * height`. The constructor
/// [`ImageView::new`] enforces this and returns `None` on mismatch.
#[derive(Copy, Clone, Debug)]
pub struct ImageView<'a> {
    /// Pixel data in row-major order. Row `y` starts at byte
    /// `y * width`.
    pub data: &'a [u8],
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
}

impl<'a> ImageView<'a> {
    /// Create a view from a raw u8 slice.
    ///
    /// Returns `None` if `data.len() != width * height`.
    pub fn new(width: usize, height: usize, data: &'a [u8]) -> Option<Self> {
        if width.checked_mul(height)? != data.len() {
            return None;
        }
        Some(Self {
            data,
            width,
            height,
        })
    }
}

/// An owned, heap-allocated grayscale image buffer (u8, row-major).
///
/// `ImageBuffer` is the owning counterpart of [`ImageView`]. It is used
/// internally by `PyramidBuffers` to hold the downsampled pyramid
/// levels and can be borrowed as an [`ImageView`] via
/// [`ImageBuffer::as_view`].
#[derive(Clone, Debug)]
pub struct ImageBuffer {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Pixel data in row-major order, zero-initialized at construction.
    /// Row `y` starts at byte `y * width`.
    pub data: Vec<u8>,
}

impl ImageBuffer {
    /// Allocate a zero-filled buffer with the given dimensions.
    ///
    /// Uses [`usize::saturating_mul`], so multiplication never wraps.
    /// Extremely large dimensions may still request an impractically
    /// large allocation.
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0; width.saturating_mul(height)],
        }
    }

    /// Borrow this buffer as a non-owning [`ImageView`].
    ///
    /// Panics if the internal invariant `data.len() == width * height`
    /// is violated, which cannot happen through the public API.
    pub fn as_view(&self) -> ImageView<'_> {
        ImageView::new(self.width, self.height, &self.data).unwrap()
    }
}
