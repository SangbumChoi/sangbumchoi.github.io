const canvas = document.querySelector("#conversation-backdrop");
if (canvas) {
  const ctx = canvas.getContext("2d");
  let frame;
  const draw = () => {
    if (!ctx) return;
    const width = window.innerWidth;
    const height = window.innerHeight;
    const ratio = Math.min(window.devicePixelRatio || 1, 1.5);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    ctx.scale(ratio, ratio);
    ctx.strokeStyle = "rgba(78, 114, 132, 0.065)";
    ctx.lineWidth = 0.65;
    const spread = Math.min(width * 0.22, 280);
    for (let i = 0; i < 22; i += 1) {
      const offset = i * 13;
      ctx.beginPath();
      ctx.moveTo(-spread + offset, height * 0.16);
      ctx.bezierCurveTo(spread * 0.6 + offset, height * 0.4, -spread + offset, height * 0.7, spread * 0.4 + offset, height * 1.15);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(width - spread * 0.2 + offset, -height * 0.2);
      ctx.bezierCurveTo(width - spread * 1.2 + offset, height * 0.25, width + spread * 0.6 + offset, height * 0.5, width - spread * 0.2 + offset, height * 0.85);
      ctx.stroke();
    }
  };
  window.addEventListener("resize", () => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(draw);
  }, { passive: true });
  draw();
}
