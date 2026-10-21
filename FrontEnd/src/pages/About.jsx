
import { Link } from "react-router-dom";

function About() {
  return (
    <main className="service-page">
      <section className="container about-page-content">
        <div className="about-page-icon">
          <i className="bi bi-eye"></i>
        </div>

        <span className="section-label">عن المشروع</span>

        <h1 className="service-page-title">
          بصيرة AI — رؤية أعمق للمعرفة
        </h1>

        <p>
          بصيرة AI مشروع يهدف إلى تقديم تجربة عربية للاستفادة
          من تقنيات الذكاء الاصطناعي في البحث عن المعلومات
          وفهمها والتحقق منها، من خلال واجهة سهلة الاستخدام.
        </p>

        <div className="about-features">
          <article>
            <i className="bi bi-chat-dots"></i>
            <h3>تجربة عربية</h3>
            <p>واجهة مصممة لتسهيل التفاعل باللغة العربية.</p>
          </article>

          <article>
            <i className="bi bi-shield-check"></i>
            <h3>التحقق من المعلومات</h3>
            <p>أدوات تساعد على فحص المعلومات وفهمها.</p>
          </article>

          <article>
            <i className="bi bi-lightbulb"></i>
            <h3>سهولة الاستخدام</h3>
            <p>خدمات متنوعة في تجربة واحدة واضحة.</p>
          </article>
        </div>

        <Link className="primary-button" to="/services">
          اكتشف خدماتنا
          <i className="bi bi-arrow-left"></i>
        </Link>
      </section>
    </main>
  );
}

export default About;
