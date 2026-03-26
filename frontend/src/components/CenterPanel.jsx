export default function CenterPanel() {
  return (
    <div className="panel">
      <div className="panel-title">TACTICAL OPTICS</div>

      <div style={{
        border: "1px solid #ffaa00",
        padding: "10px",
        marginBottom: "10px"
      }}>
        LIVE TRIAGE:
      </div>

      <div style={{
        height: "400px",
        border: "1px solid #00ff00",
        display: "flex",
        justifyContent: "center",
        alignItems: "center"
      }}>
        📹 VIDEO FEED
      </div>
    </div>
  );
}