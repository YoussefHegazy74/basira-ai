import { Component } from "react";

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  render() {
    if (this.state.hasError) {
      return (
        <main className="route-state" role="alert">
          <span className="empty-state-icon" aria-hidden="true">
            <i className="bi bi-exclamation-circle"></i>
          </span>
          <h1>حدث خطأ غير متوقع</h1>
          <p>
            تعذّر عرض هذه الصفحة. يمكنك إعادة تحميل الموقع والمحاولة مرة
            أخرى.
          </p>
          <button
            type="button"
            className="btn btn-primary"
            onClick={() => window.location.reload()}
          >
            إعادة تحميل الصفحة
          </button>
        </main>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;
