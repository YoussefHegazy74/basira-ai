import { Link } from "react-router-dom";
import BasiraLogo from "./BasiraLogo";
import { formatYear } from "../utils/format";

function Footer() {
  return (
    <footer className="site-footer">
      <div className="container footer-grid">
        <div className="footer-about">
          <Link className="brand" to="/" aria-label="بصيرة، الصفحة الرئيسية">
            <BasiraLogo size="md" />
          </Link>
          <p>
            مساحة عربية هادئة للبحث عن المعرفة وفهمها والتحقق منها، دون
            ازدحام أو ادعاءات مبالغ فيها.
          </p>
        </div>

        <nav className="footer-col" aria-label="روابط الموقع">
          <h2>الموقع</h2>
          <ul>
            <li>
              <Link to="/">الرئيسية</Link>
            </li>
            <li>
              <Link to="/services">الخدمات</Link>
            </li>
            <li>
              <Link to="/about">عن بصيرة</Link>
            </li>
          </ul>
        </nav>

        <nav className="footer-col" aria-label="روابط الأدوات">
          <h2>الأدوات</h2>
          <ul>
            <li>
              <Link to="/chat">المساعد المعرفي</Link>
            </li>
            <li>
              <Link to="/verify">التحقق من المعلومات</Link>
            </li>
            <li>
              <Link to="/tafsir">البحث في التفسير</Link>
            </li>
          </ul>
        </nav>
      </div>

      <div className="container footer-bottom">
        <p>© {formatYear()} بصيرة. جميع الحقوق محفوظة.</p>
        <a className="footer-top-link" href="#top">
          العودة إلى الأعلى
          <i className="bi bi-arrow-up" aria-hidden="true"></i>
        </a>
      </div>
    </footer>
  );
}

export default Footer;
