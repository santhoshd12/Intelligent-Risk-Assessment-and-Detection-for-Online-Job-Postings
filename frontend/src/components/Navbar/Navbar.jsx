import { Home, Search, AlertTriangle } from "lucide-react";
import './Navbar.css';

const Navbar = ({ currentPage, setCurrentPage }) => {
  return (
    <nav className="navbar">
      <div className="navbar-container">
        {/* Brand */}
        <div className="navbar-brand">
          <AlertTriangle className="navbar-icon" />
          <span className="navbar-title">JobGuard AI</span>
        </div>

        {/* Links */}
        <div className="navbar-links">
          <button
            onClick={() => setCurrentPage("home")}
            className={`nav-link ${currentPage === "home" ? "active" : ""}`}
          >
            <Home className="nav-icon" /> Home
          </button>
          <button
            onClick={() => setCurrentPage("analyze")}
            className={`nav-link ${currentPage === "analyze" ? "active" : ""}`}
          >
            <Search className="nav-icon" /> Analyze Job
          </button>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
