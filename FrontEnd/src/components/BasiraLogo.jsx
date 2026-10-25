function BasiraLogo({ size = "md", className = "" }) {
  return (
    <span
      className={`basira-logo basira-logo-${size}${
        className ? ` ${className}` : ""
      }`}
      dir="rtl"
      role="img"
      aria-label="بصيرة"
    >
      <span className="basira-logo-word" aria-hidden="true">
        بصيرة
      </span>
      <svg
        className="basira-logo-swash"
        viewBox="0 0 116 14"
        aria-hidden="true"
        focusable="false"
      >
        <path
          className="basira-swash-line"
          d="M4 8 Q22 0 40 8 T76 8 T112 8"
          fill="none"
          strokeWidth="2.5"
          strokeLinecap="round"
        />
      </svg>
    </span>
  );
}

export default BasiraLogo;
