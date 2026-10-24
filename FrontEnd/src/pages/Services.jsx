import { Link } from "react-router-dom";
import "../styles/home.css";
import { formatNumber } from "../utils/format";

const SERVICES = [
  {
    icon: "bi-chat-square-text",
    title: "المساعد المعرفي",
    description:
      "مناسب للأسئلة المفتوحة والاستفسارات القصيرة. اكتب سؤالك، واقرأ الإجابة مع إمكانية الرجوع إلى المصادر.",
    action: "ابدأ المحادثة",
    to: "/chat",
  },
  {
    icon: "bi-shield-check",
    title: "التحقق من المعلومات",
    description:
      "مناسب عندما يكون لديك نص أو معلومة جاهزة وتريد معرفة مدى دعمها لما هو موثق.",
    action: "تحقق من نص",
    to: "/verify",
  },
  {
    icon: "bi-book",
    title: "البحث في التفسير",
    description:
      "مناسب للبحث عن آية محددة وقراءة تفسيرها في صفحة واضحة بعيدة عن التشتت.",
    action: "ابحث في التفسير",
    to: "/tafsir",
  },
];

function Services() {
  return (
    <main className="page">
      <section className="marketing-hero">
        <div className="container">
          <header className="page-header" data-reveal>
            <p className="eyebrow">الأدوات</p>
            <h1>أداة مناسبة لكل سؤال</h1>
            <p>
              لا تحتاج إلى تعلم مصطلحات جديدة. اختر الطريقة الأقرب لما
              تريده، وستجد واجهة بسيطة تشرح لك الخطوة التالية.
            </p>
          </header>
        </div>
      </section>

      <section className="marketing-section" aria-labelledby="services-list-title">
        <div className="container">
          <h2 className="sr-only" id="services-list-title">
            قائمة الأدوات
          </h2>

          <div className="service-list">
            {SERVICES.map((service, index) => (
              <article className="service-row" key={service.to} data-reveal>
                <span className="service-icon" aria-hidden="true">
                  <i className={`bi ${service.icon}`}></i>
                </span>

                <div className="service-row-copy">
                  <p className="service-index">
                    الأداة {formatNumber(index + 1)}
                  </p>
                  <h2>{service.title}</h2>
                  <p>{service.description}</p>
                </div>

                <Link className="btn btn-secondary" to={service.to}>
                  {service.action}
                  <i className="bi bi-arrow-left" aria-hidden="true"></i>
                </Link>
              </article>
            ))}
          </div>

          <div className="helper-panel" data-reveal>
            <div>
              <strong>لست متأكدًا من الأداة المناسبة؟</strong>
              <p>
                ابدأ بمحادثة عادية. يمكنك وصف ما تبحث عنه، ثم اختيار أداة
                أخرى إن احتجت.
              </p>
            </div>
            <Link className="btn btn-primary" to="/chat">
              اسأل بصيرة
              <i className="bi bi-arrow-left" aria-hidden="true"></i>
            </Link>
          </div>
        </div>
      </section>

      <section className="marketing-section section-muted" aria-labelledby="services-notes-title">
        <div className="container">
          <div className="section-header" data-reveal>
            <p className="eyebrow">قبل أن تبدأ</p>
            <h2 id="services-notes-title">ثلاث ملاحظات تجعل التجربة أوضح</h2>
          </div>

          <div className="feature-grid">
            <article className="feature-card" data-reveal>
              <i className="bi bi-pencil" aria-hidden="true"></i>
              <h3>اكتب التفاصيل المهمة</h3>
              <p>
                كلما كان السؤال أو النص أوضح، كان من الأسهل فهم المطلوب
                ومراجعة النتيجة.
              </p>
            </article>

            <article className="feature-card" data-reveal>
              <i className="bi bi-journal-text" aria-hidden="true"></i>
              <h3>راجع المرجع بنفسك</h3>
              <p>
                النتيجة مساعدة للفهم، وليست بديلًا عن قراءة المصدر أو سؤال
                أهل الاختصاص.
              </p>
            </article>

            <article className="feature-card" data-reveal>
              <i className="bi bi-hourglass-split" aria-hidden="true"></i>
              <h3>امنحها وقتًا كافيًا</h3>
              <p>
                قد يستغرق تجهيز الإجابة لحظات. ستظهر حالة واضحة أثناء
                الانتظار، ويمكنك المحاولة مجددًا عند الحاجة.
              </p>
            </article>
          </div>
        </div>
      </section>
    </main>
  );
}

export default Services;
