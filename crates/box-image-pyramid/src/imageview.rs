/// Minimal borrowed grayscale image view (u8, row-major).
#[derive(Copy, Clone, Debug)]
pub struct ImageView<'a> {
    pub data: &'a [u8],
    pub width: usize,
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

/// Owned grayscale image buffer (u8).
#[derive(Clone, Debug)]
pub struct ImageBuffer {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl ImageBuffer {
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0; width.saturating_mul(height)],
        }
    }

    pub fn as_view(&self) -> ImageView<'_> {
        ImageView::new(self.width, self.height, &self.data).unwrap()
    }
}
