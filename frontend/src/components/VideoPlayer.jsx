import { useState } from "react";

export default function VideoPlayer() {
  const [source, setSource] = useState("");

  return (
    <div className="panel">
      <div className="panel-title">TACTICAL VIDEO</div>

      <input
        value={source}
        onChange={(e) => setSource(e.target.value)}
        placeholder="0 or IP stream"
      />

      <button className="btn green">CONNECT</button>

      <div style={{
        height: "300px",
        border: "1px solid #00ff00",
        display: "flex",
        alignItems: "center",
        justifyContent: "center"
      }}>
        VIDEO FEED
      </div>
    </div>
  );
}