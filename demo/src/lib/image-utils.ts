// Decode an image file/blob into RGBA pixels for the detector.
//
// The chess-corners WASM detector accepts RGBA directly (`detectRgba`) and
// converts to grayscale internally, so the demo only carries the RGBA buffer.

export interface ImageData {
  rgba: Uint8Array;
  width: number;
  height: number;
}

export async function decodeRgba(blob: Blob): Promise<ImageData> {
  const bitmap = await createImageBitmap(blob);
  try {
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Failed to get 2D context");
    ctx.drawImage(bitmap, 0, 0);
    const raw = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
    return { rgba: new Uint8Array(raw.data), width: bitmap.width, height: bitmap.height };
  } finally {
    bitmap.close();
  }
}
