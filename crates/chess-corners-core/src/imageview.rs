/// Minimal grayscale view for refinement without taking a dependency on `image`.
#[derive(Copy, Clone, Debug)]
pub struct ImageView<'a> {
    pub data: &'a [u8],
    pub width: usize,
    pub height: usize,
    /// Origin of the view in the coordinate system of the response map / base image.
    ///
    /// Use [`Self::origin`] to read this value from outside the crate. Setting the
    /// origin is done through [`Self::with_origin`], which preserves invariants.
    pub(crate) origin: [i32; 2],
}

impl<'a> ImageView<'a> {
    pub fn from_u8_slice(width: usize, height: usize, data: &'a [u8]) -> Option<Self> {
        if width.checked_mul(height)? != data.len() {
            return None;
        }
        Some(Self {
            data,
            width,
            height,
            origin: [0, 0],
        })
    }

    pub fn with_origin(
        width: usize,
        height: usize,
        data: &'a [u8],
        origin: [i32; 2],
    ) -> Option<Self> {
        Self::from_u8_slice(width, height, data).map(|mut view| {
            view.origin = origin;
            view
        })
    }

    /// Return the view's origin in the coordinate system of the response
    /// map / base image. Use [`Self::with_origin`] to construct a view
    /// with a non-zero origin.
    #[inline]
    pub fn origin(&self) -> [i32; 2] {
        self.origin
    }

    #[inline]
    pub fn supports_patch(&self, cx: i32, cy: i32, radius: i32) -> bool {
        if self.width == 0 || self.height == 0 {
            return false;
        }

        let gx = cx + self.origin[0];
        let gy = cy + self.origin[1];
        let min_x = 0;
        let min_y = 0;
        let max_x = self.width as i32 - 1;
        let max_y = self.height as i32 - 1;
        gx - radius >= min_x && gy - radius >= min_y && gx + radius <= max_x && gy + radius <= max_y
    }

    #[inline]
    pub fn sample(&self, gx: i32, gy: i32) -> f32 {
        if self.width == 0 || self.height == 0 {
            return 0.0;
        }
        let gx = gx + self.origin[0];
        let gy = gy + self.origin[1];
        let lx = gx.clamp(0, self.width.saturating_sub(1) as i32) as usize;
        let ly = gy.clamp(0, self.height.saturating_sub(1) as i32) as usize;
        self.data[ly * self.width + lx] as f32
    }

    /// Bilinear sample at subpixel coordinates. Coordinates are in the
    /// view's external frame (same as [`Self::sample`]): `origin` is
    /// applied, then the sample is clamped to the valid pixel range.
    #[inline]
    pub fn sample_bilinear(&self, gx: f32, gy: f32) -> f32 {
        if self.width == 0 || self.height == 0 {
            return 0.0;
        }

        let fx = gx + self.origin[0] as f32;
        let fy = gy + self.origin[1] as f32;

        let max_x = self.width.saturating_sub(1) as i32;
        let max_y = self.height.saturating_sub(1) as i32;

        let x0 = (fx.floor() as i32).clamp(0, max_x);
        let y0 = (fy.floor() as i32).clamp(0, max_y);
        let x1 = (x0 + 1).clamp(0, max_x);
        let y1 = (y0 + 1).clamp(0, max_y);

        // Fractional parts, guarded against samples outside the clamped
        // range (where we already snapped to the border pixel).
        let tx = (fx - x0 as f32).clamp(0.0, 1.0);
        let ty = (fy - y0 as f32).clamp(0.0, 1.0);

        let w = self.width;
        let i00 = self.data[y0 as usize * w + x0 as usize] as f32;
        let i10 = self.data[y0 as usize * w + x1 as usize] as f32;
        let i01 = self.data[y1 as usize * w + x0 as usize] as f32;
        let i11 = self.data[y1 as usize * w + x1 as usize] as f32;

        let a = i00 + (i10 - i00) * tx;
        let b = i01 + (i11 - i01) * tx;
        a + (b - a) * ty
    }
}
