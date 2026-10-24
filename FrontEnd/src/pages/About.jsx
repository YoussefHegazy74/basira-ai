import { Link } from "react-router-dom";
import "../styles/home.css";

function About() {
  return (
    <main className="page">
      <section className="marketing-hero">
        <div className="container">
          <header className="page-header" data-reveal>
            <p className="eyebrow">عن بصيرة</p>
            <h1>نؤمن أن المعرفة تبدأ بسؤال جيد</h1>
            <p>
              بصيرة مساحة عربية تجمع أدوات البحث والفهم والتحقق في واجهة
              هادئة. هدفها أن تقرّب المسافة بين سؤالك والمعلومة التي
              تستحق ثقتك.
            </p>
          </header>
        </div>
      </section>

      <section className="marketing-section" aria-labelledby="about-why-title">
        <div className="container marketing-split">
          <div data-reveal>
            <p className="eyebrow">الفكرة</p>
            <h2 id="about-why-title">لماذا وُجدت بصيرة؟</h2>
          </div>

          <div className="prose" data-reveal>
            <p>
              صار الوصول إلى المعلومات أسهل من أي وقت مضى، لكن التمييز بين
              ما يفيد وما يضلّل أصبح أكثر صعوبة. كثير من الإجابات تبدو
              واثقة، بينما السياق والمصدر هما ما يمنحها قيمتها.
            </p>
            <p>
              جاءت بصيرة لتمنح المستخدم العربي مساحة أوضح: سؤال بلغة
              طبيعية، إجابة مفهومة، وإشارة إلى ما يمكن مراجعته. لا نريد أن
              نستبدل حكمك، بل أن نساعدك على الوصول إلى قرار أكثر وعيًا.
            </p>
          </div>
        </div>
      </section>

      <section className="marketing-section section-muted" aria-labelledby="about-values-title">
        <div className="container">
          <div className="section-header" data-reveal>
            <p className="eyebrow">ما نحرص عليه</p>
            <h2 id="about-values-title">ثلاث قيم تحدد شكل التجربة</h2>
          </div>

          <div className="feature-grid">
            <article className="feature-card" data-reveal>
              <i className="bi bi-layout-text-window" aria-hidden="true"></i>
              <h3>وضوح قبل الإبهار</h3>
              <p>
                نفضّل عنوانًا مفهومًا وزرًا واضحًا على المؤثرات البصرية
                التي تشتت الانتباه عن المهمة.
              </p>
            </article>

            <article className="feature-card" data-reveal>
              <i className="bi bi-chat-quote" aria-hidden="true"></i>
              <h3>لغة تحترم القارئ</h3>
              <p>
                نكتب بالعربية الفصحى المبسطة، ونتجنب العبارات الآلية
                والوعود المبالغ فيها.
              </p>
            </article>

            <article className="feature-card" data-reveal>
              <i className="bi bi-link-45deg" aria-hidden="true"></i>
              <h3>المصدر جزء من الإجابة</h3>
              <p>
                نعرض ما يساعدك على المراجعة، ونترك لك مساحة لقراءة السياق
                كاملًا.
              </p>
            </article>
          </div>
        </div>
      </section>

      <section className="marketing-section" aria-labelledby="about-tools-title">
        <div className="container marketing-split">
          <div data-reveal>
            <p className="eyebrow">ما ستجده هنا</p>
            <h2 id="about-tools-title">أدوات مترابطة، لا خدمات متفرقة</h2>
          </div>

          <div className="prose" data-reveal>
            <p>
              يمكنك أن تبدأ بسؤال، ثم تستخدم أداة التحقق لفحص نص لديك، أو
              تنتقل إلى التفسير للبحث عن آية محددة. تجمع هذه الأدوات لغة
              التصميم نفسها، حتى تشعر أنك تنتقل داخل منتج واحد.
            </p>
            <div className="cluster">
              <Link className="btn btn-primary" to="/services">
                استكشف الأدوات
                <i className="bi bi-arrow-left" aria-hidden="true"></i>
              </Link>
              <Link className="btn btn-secondary" to="/chat">
                جرّب المحادثة
              </Link>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}

export default About;
