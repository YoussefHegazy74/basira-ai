import { Link } from "react-router-dom";
import BasiraLogo from "./BasiraLogo";

function AuthShell({
  eyebrow,
  title,
  description,
  children,
  footer,
  backTo = "/",
  backLabel = "العودة إلى الرئيسية",
  asideTitle = "مساحة عربية هادئة للفهم والتحقق",
  asideText = "بصيرة تجمع أدوات المعرفة في تجربة واحدة واضحة، تحترم لغتك وتمنحك مساحة للتفكير قبل الاعتماد على أي معلومة.",
}) {
  return (
    <main className="auth-page page">
      <div className="container auth-layout">
        <aside className="auth-aside" aria-label="عن بصيرة">
          <Link className="brand auth-brand" to="/">
            <BasiraLogo size="md" />
          </Link>

          <div className="auth-aside-copy">
            <p className="eyebrow">معرفة بلا ضجيج</p>
            <h2>{asideTitle}</h2>
            <p>{asideText}</p>
          </div>

          <ul className="auth-points">
            <li>
              <i className="bi bi-check2" aria-hidden="true"></i>
              إجابات عربية مع إمكانية الرجوع إلى المصادر
            </li>
            <li>
              <i className="bi bi-check2" aria-hidden="true"></i>
              أدوات للتحقق والبحث في التفسير
            </li>
            <li>
              <i className="bi bi-check2" aria-hidden="true"></i>
              واجهة هادئة تعمل في الوضعين النهاري والليلي
            </li>
          </ul>
        </aside>

        <section className="auth-card" aria-labelledby="auth-title">
          <Link className="auth-back" to={backTo}>
            <i className="bi bi-arrow-right" aria-hidden="true"></i>
            {backLabel}
          </Link>

          <div className="auth-heading">
            {eyebrow ? <p className="eyebrow">{eyebrow}</p> : null}
            <h1 id="auth-title">{title}</h1>
            <p>{description}</p>
          </div>

          {children}

          {footer ? <div className="auth-switch">{footer}</div> : null}
        </section>
      </div>
    </main>
  );
}

export default AuthShell;
