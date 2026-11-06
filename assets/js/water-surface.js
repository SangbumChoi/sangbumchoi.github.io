/* Continuous damped ribbons, with no circular drop impulses. */
(() => {
  const surface = document.querySelector('#water-surface');
  if (!surface) return;
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  surface.append(canvas);
  document.body.classList.add('has-water');
  const motion = matchMedia('(prefers-reduced-motion: reduce)');
  let width, height, frame = 0, previous = 0, lastInput = -10000;
  const pointer = { x: -1000, y: 0, tx: -1000, ty: 0, strength: 0 };
  const nodes = Array.from({ length: 2 }, () =>
    Array.from({ length: 97 }, () => ({ offset: 0, velocity: 0 })));

  function draw() {
    ctx.clearRect(0, 0, width, height);
    const spread = Math.min(width * 0.22, 280);
    for (let side = 0; side < 2; side++) {
      ctx.save();
      const count = width < 600 ? 15 : 32;
      const fade = ctx.createLinearGradient(side ? width : 0, 0, side ? width - spread : spread, 0);
      fade.addColorStop(0, 'rgba(78,114,132,0.22)');
      fade.addColorStop(0.45, 'rgba(78,114,132,0.16)');
      fade.addColorStop(0.8, 'rgba(78,114,132,0.045)');
      fade.addColorStop(1, 'rgba(78,114,132,0)');
      for (let line = 0; line < count; line++) {
        ctx.beginPath();
        const layer = line / (count - 1);
        for (let j = 0; j < 97; j++) {
          const t = j / 96, y = t * (height + 120) - 60;
          // A shared flowing contour fans out into unevenly spaced, fading filaments.
          const phase = t * Math.PI * (side ? 1.85 : 1.45) + (side ? 1.3 : -0.9);
          const envelope = Math.pow(Math.sin(Math.PI * t), 2);
          const base = spread * (-0.2 + 0.83 * Math.pow(layer, 1.35) +
            0.3 * Math.sin(phase) + 0.1 * Math.sin(phase * 1.7 + layer * 2.1) * envelope);
          const bend = nodes[side][j].offset * (0.25 + 0.75 * Math.sin(layer * Math.PI)) * envelope;
          const x = side ? width - base - bend : base + bend;
          if (!j) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = fade;
        ctx.globalAlpha = 0.45 + 0.55 * Math.sin(layer * Math.PI);
        ctx.lineWidth = line % 5 === 0 ? 0.95 : 0.65;
        ctx.stroke();
      }
      ctx.restore();
    }
  }
  function tick(now) {
    frame = 0;
    const dt = Math.min((now - (previous || now - 16)) / 1000, 0.032);
    previous = now;
    const ease = 1 - Math.exp(-dt * 7);
    pointer.x += (pointer.tx - pointer.x) * ease;
    pointer.y += (pointer.ty - pointer.y) * ease;
    const active = !motion.matches && now - lastInput < 350;
    pointer.strength += ((active ? 1 : 0) - pointer.strength) * (1 - Math.exp(-dt * 3));
    const spread = Math.min(width * 0.22, 280);
    let energy = pointer.strength;
    nodes.forEach((row, side) => row.forEach((node, j) => {
      const y = j / 96 * (height + 120) - 60;
      const distance = side ? width - pointer.x : pointer.x;
      const influence = Math.exp(-Math.pow((y - pointer.y) / 155, 2)) *
        Math.exp(-Math.pow((distance - spread * 0.45) / Math.max(1, spread * 0.65), 2));
      const target = 46 * influence * pointer.strength;
      // Overdamped springs return smoothly without repeated oscillations.
      node.velocity += ((target - node.offset) * 38 - node.velocity * 13) * dt;
      node.offset += node.velocity * dt;
      energy += Math.abs(node.velocity) + Math.abs(node.offset);
    }));
    draw();
    if (energy > 0.02 && !document.hidden && !motion.matches) frame = requestAnimationFrame(tick);
    else previous = 0;
  }
  function wake() {
    if (!frame && !document.hidden && !motion.matches) frame = requestAnimationFrame(tick);
  }
  function resize() {
    width = innerWidth; height = innerHeight;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
    canvas.style.width = width + 'px'; canvas.style.height = height + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    draw();
  }
  window.addEventListener('pointermove', event => {
    if (event.pointerType === 'touch' || motion.matches) return;
    if (pointer.strength < 0.001) { pointer.x = event.clientX; pointer.y = event.clientY; }
    pointer.tx = event.clientX; pointer.ty = event.clientY;
    const edge = Math.min(width * 0.22, 280);
    lastInput = event.clientX < edge || event.clientX > width - edge ? performance.now() : -10000;
    wake();
  }, { passive: true });
  window.addEventListener('resize', resize, { passive: true });
  document.documentElement.addEventListener('pointerleave', () => { lastInput = -10000; wake(); });
  function reset() {
    cancelAnimationFrame(frame); frame = 0; previous = 0; lastInput = -10000; pointer.strength = 0;
    nodes.flat().forEach(node => { node.offset = 0; node.velocity = 0; });
    draw();
  }
  motion.addEventListener('change', reset);
  document.addEventListener('visibilitychange', reset);
  resize();
})();
