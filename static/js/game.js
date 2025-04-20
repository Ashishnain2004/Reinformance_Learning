const canvas = document.getElementById("gameCanvas");
const ctx = canvas.getContext("2d");
const WINDOW_WIDTH = 360;
const WINDOW_HEIGHT = 360;

async function step() {
  const resp = await fetch("/step");
  const { pixels, reward, step: t, epsilon, q_max, fps } = await resp.json();

  const imgData = ctx.createImageData(WINDOW_WIDTH, WINDOW_HEIGHT);
  let ptr = 0;
  for (let y = 0; y < pixels.length; y++) {
    for (let x = 0; x < pixels[0].length; x++) {
      const [r, g, b] = pixels[y][x];
      imgData.data[ptr++] = r;
      imgData.data[ptr++] = g;
      imgData.data[ptr++] = b;
      imgData.data[ptr++] = 255;
    }
  }
  ctx.putImageData(imgData, 0, 0);

  // Overlay info
  ctx.fillStyle = "white";
  ctx.font = "16px Arial";
  ctx.fillText(`step ${t}`, 30, 30);
  ctx.fillText(`epsilon ${epsilon.toFixed(3)}`, 30, 50);
  ctx.fillText(`q_max ${q_max.toFixed(3)}`, 30, 70);
  ctx.fillText(`reward ${reward}`, 30, 90);
  ctx.fillText(`FPS ${fps}`, 30, 110);

  requestAnimationFrame(step);
}

document.addEventListener("DOMContentLoaded", () => {
  step();
});
