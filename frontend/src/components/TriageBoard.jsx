import { motion } from "framer-motion";

export default function TriageBoard({ persons }) {
  return (
    <div style={{ width: "300px" }}>
      <h3>🚑 TRIAGE PRIORITY</h3>

      {persons.map((p, index) => (
        <motion.div
          key={p.id}
          layout
          style={{
            padding: "10px",
            margin: "5px",
            background: p.status === "Injured" ? "#ff3333" : "#111",
          }}
        >
          #{index + 1} | {p.id} | Score: {p.score}
        </motion.div>
      ))}
    </div>
  );
}