import { Link } from "react-router-dom";
import "../styles/home.css";
import { formatNumber } from "../utils/format";

const SERVICES = [
  {
    icon: "bi-chat-square-text",
    title: "المساعد المعرفي",
    description:
      "اطرح سؤالك الشرعي بلغة عادية، واقرأ إجابة واضحة مع إمكانية الرجوع إلى ما استندت إليه.",
    action: "ابدأ المحادثة",
    to: "/chat",
  },
  {
    icon: "bi-shield-check",
    title: "التحقق من المعلومات",
    description:
      "ضع نصًا أو معلومة أمامك، واعرف ما إذا كانت المصادر تدعمها أم تحتاج إلى مراجعة.",
    action: "تحقق من نص",
    to: "/verify",
  },
  {
    icon: "bi-book",
    title: "البحث في التفسير",
    description:
      "ابحث عن آية واقرأ تفسيرها في صفحة مباشرة، بعيدًا عن التعقيد والتشتت.",
    action: "ابحث في التفسير",
    to: "/tafsir",
  },
];

const PRINCIPLES = [
  {
    icon: "bi-translate",
    title: "عربية من البداية",
    description:
      "كل الواجهة مصممة للقارئ العربي؛ لغة واضحة، واتجاه صحيح، وتفاصيل تناسب النص العربي.",
  },
  {
    icon: "bi-journal-check",
    title: "المصدر قبل الحكم",
    description:
      "الإجابة ليست نهاية الطريق. نساعدك على رؤية المرجع، ومقارنة السياق، واتخاذ قرارك بنفسك.",
  },
  {
    icon: "bi-circle-half",
    title: "هدوء بصري",
    description:
      "لا مؤثرات صاخبة ولا ألوان مبالغ فيها. مساحة مرتبة تترك التركيز للمحتوى الذي تقرأه.",
  },
];

function Home() {
  return (
    <main className="home-page">
      <section className="home-hero">
        <div className="container home-hero-grid">
          <div className="hero-copy" data-reveal>
            <p className="eyebrow">مساحة عربية للمعرفة</p>

            <h1>
              اعرف أكثر،
              <span>وثق بما تعرف.</span>
            </h1>

            <p className="lead">
              بصيرة تجمع المحادثة والتحقق من المعلومات والتفسير في تجربة
              واحدة هادئة. تعطيك الإجابة، وتذكّرك دائمًا بأن ترجع إلى
              مصدرها.
            </p>

            <div className="hero-actions">
              <Link className="btn btn-primary" to="/chat">
                ابدأ محادثة
                <i className="bi bi-arrow-left" aria-hidden="true"></i>
              </Link>
              <Link className="btn btn-secondary" to="/services">
                تعرّف على الأدوات
              </Link>
            </div>

            <ul className="hero-notes">
              <li>
                <i className="bi bi-check2" aria-hidden="true"></i>
                واجهة عربية بالكامل
              </li>
              <li>
                <i className="bi bi-check2" aria-hidden="true"></i>
                مصادر يمكن الرجوع إليها
              </li>
              <li>
                <i className="bi bi-check2" aria-hidden="true"></i>
                وضع ليلي مريح
              </li>
            </ul>
          </div>

          <div className="demo-stage" data-reveal>
            <div className="demo-card">
              <div className="demo-card-head">
                <span>مثال توضيحي</span>
                <span>
                  <i className="bi bi-chat-square-text" aria-hidden="true"></i>
                  محادثة
                </span>
              </div>

              <p className="demo-question">
                ما تفسير قوله تعالى: ﴿إِنَّ مَعَ الْعُسْرِ يُسْرًا﴾؟
              </p>

              <div className="demo-answer">
                <strong>بصيرة</strong>
                الآية تبشّر المؤمن بأن مع العسر يسرًا، وتدعوه إلى الصبر
                والرجاء. يعرض لك بصيرة التفسير والسياق من المصادر المتاحة،
                ويمكنك الرجوع إليها.
              </div>

              <div className="demo-source">
                <span className="demo-source-icon" aria-hidden="true">
                  <i className="bi bi-journal-text"></i>
                </span>
                <span className="demo-source-copy">
                  <strong>المصدر</strong>
                  <span>تفسير — سورة الشرح</span>
                </span>
              </div>

              <p className="demo-card-foot">
                <i className="bi bi-info-circle" aria-hidden="true"></i>
                مثال يوضح شكل الإجابة من المصادر، وليس إجابة فعلية.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="section" aria-labelledby="home-services-title">
        <div className="container">
          <div className="section-header" data-reveal>
            <p className="eyebrow">الأدوات</p>
            <h2 id="home-services-title">ثلاث طرق للوصول إلى المعرفة</h2>
            <p>
              اختر الأداة بحسب سؤالك. كلها تعمل بالعربية، وتترك لك مساحة
              للمراجعة قبل الوصول إلى حكم.
            </p>
          </div>

          <div className="cards-grid">
            {SERVICES.map((service, index) => (
              <Link
                className="service-card"
                to={service.to}
                key={service.to}
                data-reveal
              >
                <div className="service-card-top">
                  <span className="service-icon" aria-hidden="true">
                    <i className={`bi ${service.icon}`}></i>
                  </span>
                  <span className="service-index">
                    {formatNumber(index + 1).padStart(2, "٠")}
                  </span>
                </div>

                <h3>{service.title}</h3>
                <p>{service.description}</p>

                <span className="service-link">
                  {service.action}
                  <i className="bi bi-arrow-left" aria-hidden="true"></i>
                </span>
              </Link>
            ))}
          </div>
        </div>
      </section>

      <section className="section section-muted" aria-labelledby="home-principles-title">
        <div className="container">
          <div className="section-header" data-reveal>
            <p className="eyebrow">لماذا بصيرة؟</p>
            <h2 id="home-principles-title">أقل ادعاءً، ومساحة أكبر للفهم</h2>
            <p>
              لا نحاول أن نبدو أذكى منك. نحاول فقط أن نجعل الوصول إلى
              المعلومة ومراجعتها أسهل وأكثر ترتيبًا.
            </p>
          </div>

          <div className="principles-grid">
            {PRINCIPLES.map((principle) => (
              <article className="principle" key={principle.title} data-reveal>
                <span className="principle-icon" aria-hidden="true">
                  <i className={`bi ${principle.icon}`}></i>
                </span>
                <h3>{principle.title}</h3>
                <p>{principle.description}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="section" aria-labelledby="home-steps-title">
        <div className="container">
          <div className="section-header" data-reveal>
            <p className="eyebrow">كيف تستخدمها؟</p>
            <h2 id="home-steps-title">خطوات قليلة، ونتيجة أوضح</h2>
          </div>

          <div className="steps-grid">
            <article className="step" data-reveal>
              <span className="step-number">{formatNumber(1)}</span>
              <h3>اكتب سؤالك</h3>
              <p>
                استخدم لغتك الطبيعية. لا تحتاج إلى صياغة خاصة أو مصطلحات
                تقنية.
              </p>
            </article>

            <article className="step" data-reveal>
              <span className="step-number">{formatNumber(2)}</span>
              <h3>راجع الإجابة</h3>
              <p>
                اقرأ النص كاملًا، وانتبه إلى ما تدعمه المصادر وما لا تدعمه.
              </p>
            </article>

            <article className="step" data-reveal>
              <span className="step-number">{formatNumber(3)}</span>
              <h3>ارجع إلى المرجع</h3>
              <p>
                افتح المصدر عند الحاجة، وقارن السياق قبل أن تعتمد على أي
                معلومة.
              </p>
            </article>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="container">
          <div className="cta-panel" data-reveal>
            <div>
              <p className="eyebrow">خطوة أولى بسيطة</p>
              <h2>لديك سؤال الآن؟</h2>
              <p>
                ابدأ محادثة قصيرة، أو استخدم أداة التحقق إذا كان النص أمامك
                بالفعل.
              </p>
            </div>
            <Link className="btn btn-primary" to="/chat">
              ابدأ المحادثة
              <i className="bi bi-arrow-left" aria-hidden="true"></i>
            </Link>
          </div>
        </div>
      </section>
    </main>
  );
}

export default Home;
