export default function AlertsBanner({ alert }) {
  return (
    <div style={{
      background: alert === "EMERGENCY" ? "red" :
                 alert === "WARNING" ? "orange" : "transparent",
      padding: "5px"
    }}>
      STATUS: {alert}
    </div>
  );
}