import { useState } from "react";
import Navbar from "./components/Navbar/Navbar";
import LandingPage from "./components/LandingPage/LandingPage";
import HomePage from "./components/HomePage/HomePage";
import AnalysisPage from "./components/AnalysisPage/AnalysisPage";
import ResultsPage from "./components/ResultPage/ResultsPage";

const App = () => {
  const [currentPage, setCurrentPage] = useState("landing"); // start from landing
  const [jobText, setJobText] = useState(""); // single text field
  const [prediction, setPrediction] = useState(null);

  // Example stats
  const trainingStats = {
    totalSamples: 17880,
    realJobs: 16570,
    fakeJobs: 1310,
    accuracy: 94.2,
  };

  const distributionData = [
    { name: "Real Jobs", value: trainingStats.realJobs, color: "#22c55e" },
    { name: "Fake Jobs", value: trainingStats.fakeJobs, color: "#ef4444" },
  ];

  const modelMetrics = [
    { metric: "Accuracy", value: 94.2 },
    { metric: "Precision", value: 91.8 },
    { metric: "Recall", value: 89.5 },
    { metric: "F1-Score", value: 90.6 },
  ];

  const featureImportance = [
    { feature: "Company Profile", importance: 0.23 },
    { feature: "Job Description", importance: 0.19 },
    { feature: "Requirements", importance: 0.16 },
  ];

  // Send pasted job text to backend and get prediction
  const analyzeJob = async () => {
    try {
      if (!jobText.trim()) {
        alert("Please paste a job posting before analyzing.");
        return;
      }

      const payload = { text: jobText };

      console.log("Sending to backend:", payload);

      const response = await fetch("http://localhost:5000/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        const errorData = await response.json();
        console.error("Backend error:", errorData);
        alert("Backend error: " + (errorData.error || "Unknown error"));
        return;
      }

      const result = await response.json();
      setPrediction(result);
      setCurrentPage("results");
    } catch (error) {
      console.error("Error while analyzing job:", error);
      alert("Failed to analyze the job. Try again.");
    }
  };

  return (
    <div>
      <Navbar currentPage={currentPage} setCurrentPage={setCurrentPage} />

      {currentPage === "landing" && (
        <LandingPage setCurrentPage={setCurrentPage} />
      )}

      {currentPage === "home" && (
        <HomePage
          trainingStats={trainingStats}
          distributionData={distributionData}
          modelMetrics={modelMetrics}
          featureImportance={featureImportance}
        />
      )}

      {currentPage === "analyze" && (
        <AnalysisPage
          jobText={jobText}
          setJobText={setJobText}
          analyzeJob={analyzeJob}
        />
      )}

      {currentPage === "results" && (
        <ResultsPage
          prediction={prediction}
          jobText={jobText}
          setCurrentPage={setCurrentPage}
          setPrediction={setPrediction}
        />
      )}
    </div>
  );
};

export default App;
