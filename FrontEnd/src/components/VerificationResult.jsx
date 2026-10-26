import { verificationView } from "../utils/verification";
import "../styles/verification.css";

function Badge({ tone = "", children }) {
  if (!children) return null;
  return (
    <span className={`result-badge ${tone || ""}`.trim()}>{children}</span>
  );
}

function Assessment({ label, tone, value, children }) {
  if (!value && !children) return null;
  return (
    <div className="post-assessment">
      <div className="post-assessment-head">
        <h4>{label}</h4>
        <Badge tone={tone}>{value}</Badge>
      </div>
      {children}
    </div>
  );
}

function EvidenceList({ items }) {
  return (
    <div className="verification-evidence-list">
      {items.map((item, index) => (
        <article
          className="verification-evidence-item"
          key={`${item.reference || "evidence"}-${index}`}
        >
          <header>
            <strong dir="auto">{item.reference || "مصدر"}</strong>
            <span>{item.sourceLabel || item.source || "مصدر"}</span>
          </header>
          <p dir="auto">{item.text}</p>
        </article>
      ))}
    </div>
  );
}

function EvidenceDetails({ items, collapsible = true }) {
  if (!items?.length) return null;
  if (!collapsible) {
    return (
      <section className="verification-evidence">
        <div className="verification-evidence-summary">
          <span className="verification-evidence-title">
            <i className="bi bi-journal-text" aria-hidden="true"></i>
            الأدلة والمصادر المستخدمة{" "}
            <span className="verification-evidence-count">
              {items.length}
            </span>
          </span>
        </div>
        <EvidenceList items={items} />
      </section>
    );
  }
  return (
    <details className="verification-evidence">
      <summary>
        <span className="verification-evidence-title">
          <i className="bi bi-journal-text" aria-hidden="true"></i>
          الأدلة والمصادر المستخدمة{" "}
          <span className="verification-evidence-count">
            {items.length}
          </span>
        </span>
      </summary>
      <EvidenceList items={items} />
    </details>
  );
}

function PostComponent({ component }) {
  const evidence = component.evidence;

  return (
    <article
      className={`post-component tone-${component.severityTone}`}
      aria-labelledby={`component-${component.id}-title`}
    >
      <header className="post-component-head">
        <div>
          <h3 id={`component-${component.id}-title`}>
            {component.typeLabel}
          </h3>
        </div>
        <div className="post-component-badges">
          <Badge tone={component.sourceTone}>{component.sourceLabel}</Badge>
          {component.severity !== "none" ? (
            <span className={`verification-chip tone-${component.severityTone}`}>
              خطورة {component.severity === "high" ? "عالية" : component.severity === "medium" ? "متوسطة" : "منخفضة"}
            </span>
          ) : null}
        </div>
      </header>

      {component.excerpt ? (
        <div className="post-excerpt">
          <span>النص أو الاقتباس كما ورد</span>
          <p dir="auto">{component.excerpt}</p>
        </div>
      ) : null}

      {component.claim ? (
        <div className="post-claim">
          <span>دعوى المؤلف أو شرحه</span>
          <p dir="auto">{component.claim}</p>
        </div>
      ) : null}

      {component.missingInformation.length ? (
        <div className="post-missing">
          <h4>المعلومات الناقصة للمتابعة</h4>
          <ul>
            {component.missingInformation.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}

      {component.canonical ? (
        <div className="post-canonical">
          <div className="post-canonical-head">
            <h4>{component.canonicalLabel}</h4>
          </div>
          <blockquote className="verification-verse" dir="rtl">
            {component.canonical}
          </blockquote>
          {component.reference ? (
            <p className="verification-reference" dir="auto">
              <i className="bi bi-journal-text" aria-hidden="true"></i>
              {component.reference}
            </p>
          ) : null}
        </div>
      ) : null}

      <div className="post-assessment-grid">
        <Assessment
          label="اكتمال الاقتباس والسياق"
          tone={component.completenessTone}
          value={component.completenessLabel}
        >
          {component.omittedText ? (
            <div className="post-omitted">
              <span>الجزء غير المقتبس / السياق المحيط</span>
              <p dir="auto">{component.omittedText}</p>
            </div>
          ) : null}
          {component.completenessExplanation ? (
            <p className="post-assessment-copy">
              {component.completenessExplanation}
            </p>
          ) : null}
          {component.materialityBasis ? (
            <p className="post-assessment-help">
              {component.materialityBasis}
            </p>
          ) : null}
        </Assessment>

        {component.authenticityLabel ? (
          <Assessment
            label="درجة الحديث أو ثبوت الخبر"
            tone={component.authenticityTone}
            value={component.authenticityLabel}
          >
            {component.authenticityAssessment ? (
              <p className="post-assessment-copy" dir="auto">
                {component.authenticityAssessment}
              </p>
            ) : null}
            {component.assessor || component.authenticitySource ? (
              <p className="post-assessor">
                {component.assessor ? `الناقل: ${component.assessor}` : null}
                {component.assessor && component.authenticitySource ? " — " : null}
                {component.authenticitySource}
              </p>
            ) : null}
          </Assessment>
        ) : null}

        <Assessment
          label="تقييم شرح المؤلف"
          tone={component.interpretationTone}
          value={component.interpretationLabel}
        >
          {component.interpretationExplanation ? (
            <p className="post-assessment-copy">
              {component.interpretationExplanation}
            </p>
          ) : null}
          {!component.interpretationExplanation &&
          component.interpretationHelp ? (
            <p className="post-assessment-help">
              {component.interpretationHelp}
            </p>
          ) : null}
        </Assessment>

        <Assessment
          label="هل يدعم المصدر استدلال المؤلف؟"
          tone={component.argumentTone}
          value={component.argumentLabel}
        >
          {component.argumentExplanation ? (
            <p className="post-assessment-copy">
              {component.argumentExplanation}
            </p>
          ) : null}
          {component.argumentHelp ? (
            <p className="post-assessment-help">
              {component.argumentHelp}
            </p>
          ) : null}
        </Assessment>
      </div>

      {component.correction ? (
        <div className="verification-correction">
          <h4>تصحيح مقترح — راجع المصدر</h4>
          <p>{component.correction}</p>
        </div>
      ) : null}

      {evidence.length ? (
        <EvidenceDetails items={evidence} collapsible={false} />
      ) : (
        <p className="verification-muted">
          لم يتم العثور على نص مصدر مطابق لهذا العنصر؛ النتيجة غير محسومة.
        </p>
      )}
    </article>
  );
}

function PostVerificationResult({ view }) {
  return (
    <div className={`verification-result post-result tone-${view.tone}`} dir="rtl">
      <section
        className="verification-status"
        aria-labelledby="verification-status-title"
      >
        <span className="verification-status-icon" aria-hidden="true">
          <i
            className={
              view.isError
                ? "bi bi-exclamation-octagon"
                : view.overallStatus === "verified"
                  ? "bi bi-check2-circle"
                  : view.overallStatus === "misleading" ||
                      view.overallStatus === "contradicted"
                    ? "bi bi-exclamation-triangle"
                    : "bi bi-info-circle"
            }
          ></i>
        </span>
        <div className="verification-status-copy">
          <p className="verification-eyebrow">
            نتيجة تحليل المنشور — {view.componentCount} عنصر
          </p>
          <h2 id="verification-status-title">{view.label}</h2>
          {view.explanation ? <p>{view.explanation}</p> : null}
        </div>
      </section>

      {view.isError ? (
        <div className="verification-error" role="alert">
          <i className="bi bi-wifi-off" aria-hidden="true"></i>
          <div>
            <strong>لم يكتمل التحقق</strong>
            <p>
              {view.errorMessage ||
                "تعذّر الاتصال بخدمة التحقق. لم يتم إصدار حكم على المحتوى."}
            </p>
          </div>
        </div>
      ) : null}

      {view.submitted ? (
        <section className="post-original" aria-labelledby="post-original-title">
          <div className="verification-block-head">
            <h3 id="post-original-title">المنشور كما أُرسل</h3>
            <span className="verification-hint">محفوظ دون إعادة كتابة</span>
          </div>
          <p dir="auto">{view.submitted}</p>
        </section>
      ) : null}

      {view.components.length ? (
        <div className="post-components">
          {view.components.map((component) => (
            <PostComponent component={component} key={component.id} />
          ))}
        </div>
      ) : (
        <section className="post-no-evidence" aria-labelledby="post-no-evidence-title">
          <h3 id="post-no-evidence-title">لا يوجد اقتباس محدد للمقارنة</h3>
          <p>
            لم يتم ربط عبارة أو اقتباس محدد في المنشور بنص من المصادر.
            لم تُعرض النصوص المسترجعة لأنها لا تدعم الادعاء مباشرة، وعدم
            العثور على دليل لا يعني أن الادعاء صحيح أو خاطئ.
          </p>
        </section>
      )}

      <section
        className={`verification-danger${
          view.dangerApplicable ? "" : " is-inapplicable"
        }`}
        aria-labelledby="verification-danger-title"
      >
        <div>
          <h3 id="verification-danger-title">درجة الخطورة</h3>
          <p>
            {view.dangerApplicable
              ? view.dangerLevel
              : "غير محسوبة لهذه الحالة"}
          </p>
        </div>
        <p className="verification-danger-note">
          {view.dangerMeaning ||
            "درجة الخطورة تقدير لاحتمال الفهم المضلل، وليست درجة ثقة في النتيجة."}
        </p>
      </section>
    </div>
  );
}

function LegacyVerificationResult({ view }) {
  const supportingEvidence = view.evidence.filter(
    (item) => item.role !== "canonical"
  );

  return (
    <div className={`verification-result tone-${view.tone}`} dir="rtl">
      <section
        className="verification-status"
        aria-labelledby="verification-status-title"
      >
        <span className="verification-status-icon" aria-hidden="true">
          <i
            className={
              view.isError
                ? "bi bi-exclamation-octagon"
                : view.status === "verified"
                  ? "bi bi-check2-circle"
                  : view.status === "contradicted"
                    ? "bi bi-x-octagon"
                    : "bi bi-info-circle"
            }
          ></i>
        </span>
        <div className="verification-status-copy">
          <p className="verification-eyebrow">حالة التحقق</p>
          <h2 id="verification-status-title">{view.label}</h2>
          {view.explanation ? <p>{view.explanation}</p> : null}
        </div>
      </section>

      {view.isError ? (
        <div className="verification-error" role="alert">
          <i className="bi bi-wifi-off" aria-hidden="true"></i>
          <div>
            <strong>لم يكتمل التحقق</strong>
            <p>
              {view.errorMessage ||
                "تعذّر الاتصال بخدمة التحقق. لم يتم إصدار حكم على النص."}
            </p>
          </div>
        </div>
      ) : null}

      <div className="verification-grid">
        {view.submitted ? (
          <section
            className="verification-block verification-submitted"
            aria-labelledby="verification-submitted-title"
          >
            <div className="verification-block-head">
              <h3 id="verification-submitted-title">
                {view.submissionLabel}
              </h3>
              <span className="verification-hint">كما كتبته</span>
            </div>
            <p className="verification-quote" dir="auto">
              {view.submitted}
            </p>
          </section>
        ) : null}

        {view.canonical ? (
          <section
            className="verification-block verification-canonical"
            aria-labelledby="verification-canonical-title"
          >
            <div className="verification-block-head">
              <h3 id="verification-canonical-title">
                {view.canonicalLabel}
              </h3>
              <div className="verification-badges">
                {view.matchLabel ? (
                  <span className={`result-badge ${view.tone}`}>
                    {view.matchLabel}
                  </span>
                ) : null}
                {view.completenessLabel ? (
                  <span className="verification-chip">
                    {view.completenessLabel}
                  </span>
                ) : null}
              </div>
            </div>
            <blockquote className="verification-verse" dir="rtl">
              {view.canonical}
            </blockquote>
          </section>
        ) : null}

        {view.reference ? (
          <section
            className="verification-block verification-source"
            aria-labelledby="verification-source-title"
          >
            <div className="verification-block-head">
              <h3 id="verification-source-title">{view.sourceLabel}</h3>
            </div>
            <p className="verification-reference" dir="auto">
              <i className="bi bi-journal-text" aria-hidden="true"></i>
              {view.reference}
            </p>
          </section>
        ) : null}

        <section
          className="verification-block verification-interpretation"
          aria-labelledby="verification-interpretation-title"
        >
          <div className="verification-block-head">
            <h3 id="verification-interpretation-title">تقييم التفسير</h3>
            <span
              className={`result-badge ${
                view.interpretationTone === "neutral"
                  ? ""
                  : view.interpretationTone
              }`}
            >
              {view.interpretationLabel}
            </span>
          </div>

          {!view.interpretationProvided ? (
            <p className="verification-muted">
              أضف شرحًا في المنشور إذا أردت تقييمه بشكل منفصل عن النص.
            </p>
          ) : view.interpretationStatus === "error" ? (
            <p className="verification-error-text">
              {view.interpretationError ||
                "تعذّر تقييم التفسير. هذا لا يعني أن التفسير صحيح أو خاطئ."}
            </p>
          ) : (
            <>
              {view.interpretationAssessment ? (
                <p className="verification-assessment">
                  {view.interpretationAssessment}
                </p>
              ) : (
                <p className="verification-muted">
                  لم تتوفر تفاصيل كافية لتقييم التفسير.
                </p>
              )}
              {view.interpretationCorrection ? (
                <div className="verification-correction">
                  <h4>تصحيح مقترح — راجع المصدر</h4>
                  <p>{view.interpretationCorrection}</p>
                </div>
              ) : null}
            </>
          )}
        </section>
      </div>

      {view.evidenceStatus === "relevant" && supportingEvidence.length ? (
        <EvidenceDetails items={supportingEvidence} />
      ) : (
        <p className="verification-muted">
          {view.evidenceNote ||
            (view.evidenceStatus === "not_relevant"
              ? "تم استرجاع نصوص من المصادر، لكنها لا تدعم الادعاء مباشرة؛ لم تُعرض لتجنب الإيحاء بدليل غير موجود."
              : "لم يتم العثور على نصوص مسترجعة مرتبطة بهذا الادعاء.")}{" "}
          راجع النص الكامل والمرجع أعلاه قبل الاعتماد.
        </p>
      )}

      <section
        className={`verification-danger${
          view.dangerApplicable ? "" : " is-inapplicable"
        }`}
        aria-labelledby="verification-danger-title"
      >
        <div>
          <h3 id="verification-danger-title">درجة الخطورة</h3>
          <p>
            {view.dangerApplicable
              ? view.dangerLevel
              : "غير محسوبة لهذه الحالة"}
          </p>
        </div>
        <p className="verification-danger-note">
          {view.dangerMeaning ||
            "درجة الخطورة تقدير لاحتمال الفهم المضلل، وليست درجة ثقة في النتيجة."}
        </p>
      </section>
    </div>
  );
}

function VerificationResult({ data }) {
  const view = verificationView(data);
  if (!view) return null;
  if (view.analysisMode === "post") {
    return <PostVerificationResult view={view} />;
  }
  return <LegacyVerificationResult view={view} />;
}

export default VerificationResult;
