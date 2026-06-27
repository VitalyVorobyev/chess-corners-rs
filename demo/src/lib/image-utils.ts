// Utility functions for loading images and extracting RGBA pixel data.
//
// The chess-corners WASM detector accepts RGBA directly (`detectRgba`) and
// converts to grayscale internally, so the demo only carries the RGBA buffer.

export interface ImageData {
  rgba: Uint8Array;
  width: number;
  height: number;
}

/** Read the RGBA pixels out of a decoded bitmap via an OffscreenCanvas. */
export function rgbaFromBitmap(bitmap: ImageBitmap): ImageData {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Failed to get 2D context");
  ctx.drawImage(bitmap, 0, 0);
  const raw = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
  return {
    rgba: new Uint8Array(raw.data),
    width: bitmap.width,
    height: bitmap.height,
  };
}

/** Decode a File/Blob into both an ImageBitmap (for display) and RGBA pixels. */
export async function loadImage(
  file: File | Blob,
): Promise<{ bitmap: ImageBitmap; data: ImageData }> {
  const bitmap = await createImageBitmap(file);
  return { bitmap, data: rgbaFromBitmap(bitmap) };
}
