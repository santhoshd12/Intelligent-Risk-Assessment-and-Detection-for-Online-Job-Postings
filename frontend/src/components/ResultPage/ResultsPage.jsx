import { useRef } from "react";
import TemporalAnalysis from "../Temporal_Analysis/TemporalAnalysis";
import "./ResultPage.css";

const ResultsPage = ({ prediction, setCurrentPage }) => {
  const reportRef = useRef();

  const handlePrint = () => {
    if (reportRef.current) {
      const printContents = reportRef.current.innerHTML;
      const originalContents = document.body.innerHTML;

      document.body.innerHTML = printContents;
      window.print();
      document.body.innerHTML = originalContents;
      window.location.reload(); // Restore React state
    }
  };

  return (
    <div className="results-page">
      <div className="report-section" ref={reportRef}>
        <h1 className="title">Job Analysis Result</h1>

        {prediction ? (
          <div className="prediction-result">
            <p>
              <strong>Prediction:</strong> {prediction.prediction}
            </p>
            <p>
              <strong>Confidence:</strong>{" "}
              {(prediction.probability * 100).toFixed(2)}%
            </p>
          </div>
        ) : (
          <p className="no-result">No prediction available.</p>
        )}

        {/* Temporal Analysis */}
        <TemporalAnalysis temporalData={prediction?.temporal} />
      </div>

      <div className="button-group">
        <button
          className="analyze-btn"
          onClick={() => setCurrentPage("analyze")}
        >
          Analyze Another Job
        </button>
        <button className="export-btn" onClick={handlePrint}>
          Print Report
        </button>
      </div>
    </div>
  );
};

export default ResultsPage;
