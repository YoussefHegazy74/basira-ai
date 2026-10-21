
import { Link } from "react-router-dom";

function Footer() {
  return (
    <footer className="footer">
      <div className="container footer-content">
        <Link className="brand footer-brand" to="/">
          <span className="brand-icon">
            <i className="bi bi-eye-fill"></i>
          </span>

          <span>
            بصيرة <small>Basira AI</small>
          </span>
        </Link>

        <p>
          © {new Date().getFullYear()} Basira AI — جميع الحقوق محفوظة.
        </p>

        <a
          className="back-to-top"
          href="#root"
          aria-label="العودة للأعلى"
        >
          <i className="bi bi-arrow-up"></i>
        </a>
      </div>
    </footer>
  );
}

export default Footer;
