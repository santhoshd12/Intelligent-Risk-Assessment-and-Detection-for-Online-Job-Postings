import "./LandingPage.css";

const LandingPage = ({ setCurrentPage }) => {
  return (
    <div className="landing-page">
      <header className="landing-header">
        <h1>
          Intelligent Risk Assessment and Detection System for Safeguarding Job
          Seekers Against Fraudulent Online Postings
        </h1>
        <p>
          Protect yourself from fraudulent job postings with AI-powered
          analysis. Analyze job postings by pasting the description or providing
          the job link.
        </p>
        <button className="start-btn" onClick={() => setCurrentPage("home")}>
          Get Started
        </button>
      </header>

      <section className="features-section">
        <h2>🔍 Features</h2>
        <div className="features-cards">
          <div className="feature-card">
            <h3>Text Analysis</h3>
            <p>
              Paste the job description directly and get an AI-based prediction
              on whether it's genuine or fraudulent.
            </p>
          </div>
          <div className="feature-card">
            <h3>Link Analysis</h3>
            <p>
              Provide a LinkedIn job posting URL. The system will fetch the full
              posting and run the prediction automatically.
            </p>
          </div>
          <div className="feature-card">
            <h3>Prediction & Confidence</h3>
            <p>
              Receive a clear prediction along with the confidence score,
              showing the likelihood of the job being genuine.
            </p>
          </div>
          <div className="feature-card">
            <h3>Temporal Analysis</h3>
            <p>
              See detailed temporal analysis of the posting, including patterns,
              suspicious elements, and content structure.
            </p>
          </div>
        </div>
      </section>

      <section className="tech-stack-section">
        <h2>🛠 Technology Stack</h2>
        <ul>
          <li>Frontend: React.js</li>
          <li>Backend: Python Flask</li>
          <li>Web Scraping: Selenium, BeautifulSoup</li>
          <li>Machine Learning: scikit-learn (Random Forest Classifier)</li>
          <li>NLP: spaCy, TF-IDF vectorizer</li>
          <li>PDF Export & Print: jsPDF / Browser Print</li>
        </ul>
      </section>

      <section className="workflow-section">
        <h2>⚡ How It Works</h2>
        <ol>
          <li>Select input method: job description text or job posting URL.</li>
          <li>System fetches or reads the content automatically.</li>
          <li>
            AI model predicts whether the posting is genuine or fraudulent.
          </li>
          <li>
            View the prediction along with confidence and temporal analysis.
          </li>
          <li>Export report as PDF or print directly for your records.</li>
        </ol>
      </section>

      <footer className="landing-footer">
        <p>© 2025 JobGuard AI. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default LandingPage;
