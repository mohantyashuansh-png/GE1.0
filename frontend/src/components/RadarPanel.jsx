import { useEffect, useRef } from "react";

export default function RadarPanel() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");

    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.strokeStyle = "#00ff00";
    ctx.beginPath();
    ctx.arc(150, 150, 100, 0, Math.PI * 2);
    ctx.stroke();
  }, []);

  return (
    <div className="panel">
      <div className="panel-title">RADAR</div>
      <canvas ref={canvasRef} width={300} height={300} />
    </div>
  );
}