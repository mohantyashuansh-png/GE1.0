export default function VLMControls() {
  return (
    <div className="panel">
      <div className="panel-title">AI CONTROLS</div>

      <button className="btn blue">SCAN HAZARDS</button>
      <button className="btn orange">TRIAGE VICTIM</button>

      <input placeholder="Shirt Color" />
      <input placeholder="Pants Color" />

      <button className="btn purple">HUNT VIP</button>
      <button className="btn red">CLEAR TARGET</button>
    </div>
  );
}