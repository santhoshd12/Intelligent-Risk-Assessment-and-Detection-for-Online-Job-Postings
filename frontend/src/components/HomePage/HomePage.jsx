import { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Tooltip,
  BarChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Bar,
} from "recharts";
import { BarChart3 } from "lucide-react";
import "./HomePage.css";

const HomePage = () => {
  const [trainingStats, setTrainingStats] = useState(null);
  const [distributionData, setDistributionData] = useState([]);
  const [modelMetrics, setModelMetrics] = useState([]);

  useEffect(() => {
    fetch("http://localhost:5000/api/metrics")
      .then((res) => res.json())
      .then((data) => {
        setTrainingStats({
          totalSamples: data.total_samples,
          realJobs: data.real_jobs,
          fakeJobs: data.fake_jobs,
          accuracy: data.model_accuracy.toFixed(2),
        });

        setDistributionData([
          { name: "Legitimate Jobs", value: data.real_jobs, color: "#22c55e" },
          { name: "Fraudulent Jobs", value: data.fake_jobs, color: "#ef4444" },
        ]);

        setModelMetrics([
          { metric: "True Negative", value: data.confusion_matrix[0][0] },
          { metric: "False Positive", value: data.confusion_matrix[0][1] },
          { metric: "False Negative", value: data.confusion_matrix[1][0] },
          { metric: "True Positive", value: data.confusion_matrix[1][1] },
        ]);
      })
      .catch((err) => console.error("Error fetching metrics:", err));
  }, []);

  if (!trainingStats)
    return (
      <h1 style={{ textAlign: "center", marginTop: "20%" }}>
        Loading metrics...
      </h1>
    );

  return (
    <div className="homepage">
      <div className="homepage-container">
        {/* Header */}
        <div className="homepage-header" style={{ textAlign: "center" }}>
          <h1 style={{ maxWidth: "80%", margin: "0 auto" }}>
            Intelligent Risk Assessment and Detection for Online Postings
          </h1>
          <p style={{ maxWidth: "80%", margin: "10px auto" }}>
            Advanced machine learning system to identify fraudulent job postings
            and protect job seekers from scams
          </p>
        </div>

        {/* Stats */}
        <div className="stats-grid">
          <div className="stat-card" style={{ textAlign: "center" }}>
            <h3>{trainingStats.totalSamples.toLocaleString()}</h3>
            <p>Total Training Samples</p>
          </div>
          <div className="stat-card" style={{ textAlign: "center" }}>
            <h3 className="green">{trainingStats.realJobs.toLocaleString()}</h3>
            <p>Legitimate Jobs</p>
          </div>
          <div className="stat-card" style={{ textAlign: "center" }}>
            <h3 className="red">{trainingStats.fakeJobs.toLocaleString()}</h3>
            <p>Fraudulent Jobs</p>
          </div>
          <div className="stat-card" style={{ textAlign: "center" }}>
            <h3 className="purple">{trainingStats.accuracy}%</h3>
            <p>Model Accuracy</p>
          </div>
        </div>

        {/* Charts */}
        <div className="charts-grid">
          <div className="chart-card" style={{ textAlign: "center" }}>
            <h3>
              <BarChart3 className="icon" /> Job Distribution
            </h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={distributionData}
                  dataKey="value"
                  cx="50%"
                  cy="50%"
                  outerRadius={120}
                  innerRadius={60}
                >
                  {distributionData.map((d, i) => (
                    <Cell key={i} fill={d.color} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="chart-card" style={{ textAlign: "center" }}>
            <h3>Model Performance</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={modelMetrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="metric" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="value" fill="#3b82f6" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HomePage;
