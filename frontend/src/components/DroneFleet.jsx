import { useEffect, useState } from "react";

const statuses = [
  "ALPHA: Scanning Sector 7",
  "BRAVO: Returning (12% Battery)",
  "CHARLIE: Target Locked",
];

export default function DroneFleet() {
  const [status, setStatus] = useState(statuses[0]);

  useEffect(() => {
    const interval = setInterval(() => {
      setStatus(statuses[Math.floor(Math.random() * statuses.length)]);
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <h3>🛸 DRONE FLEET</h3>
      <p>{status}</p>
    </div>
  );
}