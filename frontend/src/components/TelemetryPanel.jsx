export default function TelemetryPanel({ persons = [] }) {
  return (
    <div className="panel">
      <div className="panel-title">TELEMETRY</div>

      <div className="terminal">
        {persons.length === 0
          ? "No persons"
          : persons.map((p) => (
              <div key={p.id}>
                {p.id} | {p.status} | {p.score}
              </div>
            ))}
      </div>
    </div>
  );
}