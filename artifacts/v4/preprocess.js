// Experimental 128/224 preprocessing, same arithmetic as production.
export function pixelsToCHW(rgba, width, height, size) {
  const output = new Float32Array(3 * size * size), plane = size * size;
  for (let y = 0; y < size; y++) {
    const sy = Math.max(0, (y + .5) * height / size - .5);
    const y0 = Math.floor(sy), y1 = Math.min(y0 + 1, height - 1), wy = sy - y0;
    for (let x = 0; x < size; x++) {
      const sx = Math.max(0, (x + .5) * width / size - .5);
      const x0 = Math.floor(sx), x1 = Math.min(x0 + 1, width - 1), wx = sx - x0;
      for (let channel = 0; channel < 3; channel++) {
        const at = (px, py) => rgba[(py * width + px) * 4 + channel];
        const top = at(x0, y0) * (1 - wx) + at(x1, y0) * wx;
        const bottom = at(x0, y1) * (1 - wx) + at(x1, y1) * wx;
        output[channel * plane + y * size + x] = (top * (1 - wy) + bottom * wy) / 255;
      }
    }
  }
  return output;
}

export function imageToCHW(image,size) {
  const width=image.width,height=image.height;
  const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;
  const ctx=canvas.getContext('2d',{willReadFrequently:true,colorSpace:'srgb'});
  if(!ctx)throw Error('Canvas unavailable');
  ctx.fillStyle='#fff';ctx.fillRect(0,0,width,height);ctx.drawImage(image,0,0);
  const tensor=pixelsToCHW(ctx.getImageData(0,0,width,height).data,width,height,size);
  canvas.width=canvas.height=1;return tensor;
}
