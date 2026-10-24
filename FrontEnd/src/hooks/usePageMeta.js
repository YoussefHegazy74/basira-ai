import { useEffect } from "react";
import { useLocation } from "react-router-dom";

const PAGE_META = {
  "/": {
    title: "بصيرة | مساحة عربية للمعرفة والتحقق",
    description:
      "بصيرة منصة عربية هادئة تساعدك على البحث عن المعرفة وفهمها والتحقق من مصادرها.",
  },
  "/about": {
    title: "عن بصيرة | بصيرة",
    description:
      "تعرّف على فكرة بصيرة وطريقتها في تقديم تجربة معرفية عربية واضحة وموثوقة.",
  },
  "/services": {
    title: "الأدوات والخدمات | بصيرة",
    description:
      "استكشف أدوات بصيرة للمحادثة والتحقق من المعلومات والبحث في التفسير.",
  },
  "/chat": {
    title: "المساعد المعرفي | بصيرة",
    description:
      "اسأل باللغة العربية واحصل على إجابات واضحة مع إمكانية الرجوع إلى المصادر.",
  },
  "/verify": {
    title: "التحقق من المعلومات | بصيرة",
    description:
      "راجع نصًا أو معلومة واعرف مدى دعم المصادر لها قبل الاعتماد عليها.",
  },
  "/tafsir": {
    title: "البحث في التفسير | بصيرة",
    description:
      "ابحث عن آية واقرأ تفسيرها في تجربة عربية بسيطة وواضحة.",
  },
  "/signin": {
    title: "تسجيل الدخول | بصيرة",
    description: "سجّل دخولك لمتابعة محادثاتك وإعدادات حسابك في بصيرة.",
  },
  "/signup": {
    title: "إنشاء حساب | بصيرة",
    description: "أنشئ حسابك في بصيرة للاحتفاظ بمحادثاتك واستخدام الأدوات.",
  },
  "/forgot-password": {
    title: "استعادة كلمة المرور | بصيرة",
    description: "اطلب رابطًا آمنًا لاستعادة كلمة مرور حسابك في بصيرة.",
  },
  "/profile": {
    title: "الملف الشخصي | بصيرة",
    description: "راجع بيانات حسابك وأمانه وإعداداته في بصيرة.",
  },
  "/admin": {
    title: "لوحة الإدارة | بصيرة",
    description: "إدارة حسابات المستخدمين والاستخدام في بصيرة.",
  },
};

export function usePageMeta() {
  const { pathname } = useLocation();

  useEffect(() => {
    const meta =
      PAGE_META[pathname] ||
      (pathname.startsWith("/auth/resetpassword/")
        ? {
            title: "تعيين كلمة مرور جديدة | بصيرة",
            description: "اختر كلمة مرور جديدة لحسابك في بصيرة.",
          }
        : {
            title: "الصفحة غير موجودة | بصيرة",
            description: "تعذّر العثور على الصفحة المطلوبة في بصيرة.",
          });

    document.title = meta.title;
    const description = document.querySelector('meta[name="description"]');
    if (description) description.setAttribute("content", meta.description);
  }, [pathname]);
}
