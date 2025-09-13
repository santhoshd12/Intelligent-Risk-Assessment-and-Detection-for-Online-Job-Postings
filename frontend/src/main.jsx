import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App.jsx";
import TemporalAnalysis from "./components/Temporal_Analysis/TemporalAnalysis.jsx";

createRoot(document.getElementById("root")).render(<App />);
