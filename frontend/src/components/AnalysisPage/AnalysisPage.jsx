import { useState } from "react";
import ResultsPage from "../ResultPage/ResultsPage";
import "./AnalysisPage.css";

const AnalysisPage = () => {
  const [selectedMethod, setSelectedMethod] = useState(null);
  const [jobText, setJobText] = useState("");
  const [jobLink, setJobLink] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [jobData, setJobData] = useState(null);
  const [showResultPage, setShowResultPage] = useState(false);

  const handleAnalyze = async () => {
    setLoading(true);
    setResult(null);
    setJobData(null);

    try {
      let payload = {};
      let endpoint = "";

      if (selectedMethod === "description") {
        if (!jobText.trim()) {
          alert("Please enter the job description.");
          setLoading(false);
          return;
        }

        // Split text: first line = title, rest = description
        const lines = jobText.trim().split("\n");
        const title = lines[0].trim();
        const description = lines.slice(1).join("\n").trim() || lines[0].trim();

        payload = { title, description };
        endpoint = "http://localhost:5000/api/predict";
      } else if (selectedMethod === "link") {
        if (!jobLink.trim()) {
          alert("Please enter the job URL.");
          setLoading(false);
          return;
        }
        payload = { url: jobLink };
        endpoint = "http://localhost:5000/api/fetch-and-predict";
      }

      const response = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      console.log("Response from backend:", data);

      if (data.error) {
        alert(`Error: ${data.error}`);
        setLoading(false);
        return;
      }

      setResult(data);
      setJobData({
        description: data.description || "N/A",
        title: data.title || "N/A",
        temporal: data.temporal || null,
      });

      setShowResultPage(true);
    } catch (err) {
      console.error("Error:", err);
      alert("Something went wrong. Check console for details.");
    }

    setLoading(false);
  };

  const handleBack = () => {
    setShowResultPage(false);
    setResult(null);
    setJobData(null);
    setJobText("");
    setJobLink("");
    setSelectedMethod(null);
  };

  if (showResultPage) {
    return (
      <ResultsPage
        prediction={result}
        jobData={jobData}
        setCurrentPage={handleBack}
      />
    );
  }

  return (
    <div className="analysis-page">
      <div className="analysis-container">
        <div className="analysis-header">
          <h1>Analyze Job Posting</h1>
          <p>Select how you want to provide the job details</p>
        </div>

        {!selectedMethod && (
          <div className="cards-container">
            <div
              className="input-card"
              onClick={() => setSelectedMethod("description")}
            >
              <h3>Paste Job Description</h3>
              <p>
                Provide the full job posting text manually. First line will be
                treated as title, rest as description.
              </p>
            </div>

            <div
              className="input-card"
              onClick={() => setSelectedMethod("link")}
            >
              <h3>Check with Job Link</h3>
              <p>
                Provide a job posting URL to fetch the details automatically.
              </p>
            </div>
          </div>
        )}

        {selectedMethod === "description" && (
          <div className="analysis-box">
            <textarea
              value={jobText}
              onChange={(e) => setJobText(e.target.value)}
              rows={10}
              placeholder="Enter job title on the first line, followed by the job description..."
            />
            <button onClick={handleAnalyze} disabled={loading}>
              {loading ? "Analyzing..." : "Analyze Job Posting"}
            </button>
            <button onClick={() => setSelectedMethod(null)}>← Go Back</button>
          </div>
        )}

        {selectedMethod === "link" && (
          <div className="analysis-box">
            <input
              type="text"
              value={jobLink}
              onChange={(e) => setJobLink(e.target.value)}
              placeholder="Enter LinkedIn job posting URL..."
            />
            <button onClick={handleAnalyze} disabled={loading}>
              {loading ? "Fetching & Analyzing..." : "Fetch Job Posting"}
            </button>
            <button onClick={() => setSelectedMethod(null)}>← Go Back</button>
          </div>
        )}
      </div>
    </div>
  );
};

export default AnalysisPage;
