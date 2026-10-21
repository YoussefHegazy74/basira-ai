
import { Link } from "react-router-dom";
import "../App.css";

function Home() {
  const services = [
    {
      icon: "bi-chat-square-text",
      number: "01",
      title: "المساعد الذكي",
      description:
        "اسأل عن المعلومات التي تبحث عنها، واحصل على إجابات واضحة باللغة العربية.",
      link: "/chat",
      action: "ابدأ المحادثة",
    },
    {
      icon: "bi-shield-check",
      number: "02",
      title: "التحقق من المعلومات",
      description:
        "افحص النصوص والمعلومات، وساعد نفسك على فهم مدى موثوقيتها.",
      link: "/verify",
      action: "تحقق الآن",
    },
    {
      icon: "bi-book-half",
      number: "03",
      title: "التفسير",
      description:
        "استكشف خدمة التفسير وابحث عن الآيات والمعلومات المرتبطة بها.",
      link: "/tafsir",
      action: "اكتشف الخدمة",
    },
  ];

  return (
    <main>
      {/* Hero Section */}
      <section className="hero" id="home">
        <div className="container hero-content">
          <div className="hero-text">
            <div className="eyebrow">
              <span className="status-dot"></span>
              الذكاء الاصطناعي باللغة العربية
            </div>

            <h1>
              المعرفة تبدأ من
              <span> بصيرة.</span>
            </h1>

            <p>
              منصتك الذكية للبحث والفهم والتحقق من المعلومات.
              اكتشف تجربة عربية تساعدك على الوصول إلى المعرفة
              بطريقة أوضح وأسهل.
            </p>

            <div className="hero-actions">
              <Link className="primary-button" to="/services">
                اكتشف خدماتنا
                <i className="bi bi-arrow-left"></i>
              </Link>

              <Link className="secondary-button" to="/about">
                تعرّف علينا
              </Link>
            </div>

            <div className="hero-note">
              <i className="bi bi-stars"></i>
              <span>تجربة عربية، بسيطة وذكية</span>
            </div>
          </div>

          {/* Hero Visual */}
          <div className="hero-visual">
            <div className="visual-glow"></div>

            <div className="orbit orbit-one"></div>
            <div className="orbit orbit-two"></div>

            <div className="main-symbol">
              <i className="bi bi-eye"></i>
              <span>بصيرة</span>
              <small>رؤية أعمق للمعرفة</small>
            </div>

            <div className="floating-card floating-top">
              <span className="floating-icon">
                <i className="bi bi-stars"></i>
              </span>

              <span>
                <strong>مساعد ذكي</strong>
                <small>إجابات باللغة العربية</small>
              </span>
            </div>

            <div className="floating-card floating-bottom">
              <span className="floating-icon">
                <i className="bi bi-shield-check"></i>
              </span>

              <span>
                <strong>فهم أفضل</strong>
                <small>أدوات للبحث والتحقق</small>
              </span>
            </div>
          </div>
        </div>

        <div className="hero-decoration decoration-one"></div>
        <div className="hero-decoration decoration-two"></div>
      </section>

      {/* Services Section */}
      <section className="services section" id="services">
        <div className="container">
          <div className="section-heading">
            <span className="section-label">
              ماذا تقدم بصيرة؟
            </span>

            <h2>
              كل ما تحتاجه، <span>في مكان واحد</span>
            </h2>

            <p>
              أدوات ذكية صُممت لتجعل الوصول إلى المعلومات
              والتعامل معها أكثر سهولة.
            </p>
          </div>

          <div className="services-grid">
            {services.map(function (service) {
              return (
                <Link
                  className="service-card"
                  key={service.number}
                  to={service.link}
                  aria-label={`${service.title} - ${service.action}`}
                >
                  <div className="card-top">
                    <span className="service-icon">
                      <i className={`bi ${service.icon}`}></i>
                    </span>

                    <span className="service-number">
                      {service.number}
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
      </section>

      {/* About Section */}
      <section className="about-section" id="about">
        <div className="container about-content">
          <div className="about-icon">
            <i className="bi bi-lightbulb"></i>
          </div>

          <div>
            <span className="section-label">
              عن المشروع
            </span>

            <h2>التقنية في خدمة المعرفة</h2>

            <p>
              بصيرة AI مشروع يهدف إلى تقديم تجربة عربية
              للاستفادة من تقنيات الذكاء الاصطناعي في البحث
              عن المعلومات وفهمها والتحقق منها، من خلال
              واجهة سهلة الاستخدام.
            </p>
          </div>

          <Link className="about-button" to="/services">
            استكشف الخدمات
            <i className="bi bi-arrow-left"></i>
          </Link>
        </div>
      </section>
    </main>
  );
}

export default Home;
