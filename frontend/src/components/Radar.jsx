import { useEffect, useRef } from "react";

export default function Radar({ data }) {
  const canvasRef = useRef();
  const lastPositions = useRef({});

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // GRID
    ctx.strokeStyle = "rgba(0,255,0,0.2)";
    for (let i = 0; i < canvas.width; i += 40) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, canvas.height);
      ctx.stroke();
    }
    for (let i = 0; i < canvas.height; i += 40) {
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(canvas.width, i);
      ctx.stroke();
    }

    // COMPASS
    ctx.fillStyle = "#00ff00";
    ctx.fillText("N", canvas.width / 2, 10);
    ctx.fillText("S", canvas.width / 2, canvas.height - 10);
    ctx.fillText("E", canvas.width - 20, canvas.height / 2);
    ctx.fillText("W", 10, canvas.height / 2);

    data.active_persons?.forEach((p) => {
      const { x, y, id } = p;

      // DOT
      ctx.fillStyle = "#00ff00";
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fill();

      // VELOCITY LINE
      if (lastPositions.current[id]) {
        const dx = x - lastPositions.current[id].x;
        const dy = y - lastPositions.current[id].y;

        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = "#ffaa00";
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + dx * 5, y + dy * 5);
        ctx.stroke();
      }

      lastPositions.current[id] = { x, y };
    });
  }, [data]);

  return <canvas ref={canvasRef} width={500} height={500} />;
}