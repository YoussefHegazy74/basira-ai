
import { Link } from "react-router-dom";

const services = [
  {
    icon: "bi-chat-square-text",
    title: "المساعد الذكي",
    description:
      "اطرح أسئلتك واحصل على إجابات تساعدك على الوصول إلى المعلومات.",
    link: "/chat",
    action: "ابدأ المحادثة",
  },
  {
    icon: "bi-shield-check",
    title: "التحقق من المعلومات",
    description:
      "استخدم أدوات التحقق لفهم مدى موثوقية المعلومات والنصوص.",
    link: "/verify",
    action: "تحقق الآن",
  },
  {
    icon: "bi-book-half",
    title: "التفسير",
    description:
      "استخدم خدمة التفسير للوصول إلى المعلومات المرتبطة بالآيات.",
    link: "/tafsir",
    action: "استكشف التفسير",
  },
];

function Services() {
  return (
    <main className="service-page">
      <div className="container">
        <div className="section-heading">
          <span className="section-label">خدمات بصيرة</span>

          <h1 className="service-page-title">
            أدوات ذكية لخدمة المعرفة
          </h1>

          <p>
            استكشف خدمات بصيرة AI واختر الأداة المناسبة
            لاحتياجك.
          </p>
        </div>

        <div className="services-grid">
          {services.map(function (service, index) {
            return (
              <Link
                className="service-card"
                key={service.title}
                to={service.link}
                aria-label={`${service.title} - ${service.action}`}
              >
                <div className="card-top">
                  <span className="service-icon">
                    <i className={`bi ${service.icon}`}></i>
                  </span>

                  <span className="service-number">
                    {String(index + 1).padStart(2, "0")}
                  </span>
                </div>

                <h3>{service.title}</h3>

                <p>{service.description}</p>

                <span className="service-link">
                  {service.action}
                  <i className="bi bi-arrow-left"></i>
                </span>
              </Link>
            );
          })}
        </div>
      </div>
    </main>
  );
}

export default Services;
