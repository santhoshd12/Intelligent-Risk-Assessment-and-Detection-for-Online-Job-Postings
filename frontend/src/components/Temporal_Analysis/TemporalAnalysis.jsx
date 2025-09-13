import PostingChart from "./PostingChart";
import PostingTable from "./PostingTable";
import InsightBox from "./InsightBox";
import "./TemporalAnalysis.css";

const TemporalAnalysis = ({ temporalData }) => {
  if (
    !temporalData ||
    !temporalData.similar_jobs ||
    temporalData.similar_jobs.length === 0
  ) {
    return <p className="placeholder">No temporal analysis data available.</p>;
  }

  const chartData = temporalData.chart_data.map((item) => ({
    date: new Date(item.date).toLocaleDateString(),
    frequency: item.count,
  }));

  const tableData = temporalData.similar_jobs.map((job) => ({
    date: new Date(job.date).toLocaleDateString(),
    platform: job.company || "Unknown",
    similarity: (job.similarity * 100).toFixed(2),
    original: job.description,
    modified: job.description,
  }));

  const maxSim = temporalData.similar_jobs.length
    ? Math.max(...temporalData.similar_jobs.map((j) => j.similarity)) * 100
    : 0;

  return (
    <div className="temporal-card">
      <h3>Temporal Analysis – Frequency & Similarity</h3>
      <InsightBox
        insight={`💡 Found ${
          temporalData.similar_jobs.length
        } similar jobs in the past 7 days. Highest similarity: ${maxSim.toFixed(
          2
        )}%`}
      />
      <div className="temporal-chart">
        <PostingChart data={chartData} />
      </div>
      <div className="temporal-table">
        <PostingTable data={tableData} />
      </div>
    </div>
  );
};

export default TemporalAnalysis;
