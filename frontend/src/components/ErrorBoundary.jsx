// src/components/ErrorBoundary.jsx
import { Component } from "react";

/**
 * Top-level React error boundary. Catches render-time exceptions in its
 * subtree so a single panel crashing doesn't blank the entire dashboard.
 * Dark-themed to match the rest of the live app.
 */
export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error("ErrorBoundary caught an error:", error, info);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="rounded-xl border border-red-800 bg-red-950/30 p-6 text-center">
          <p className="text-sm font-semibold text-red-300">
            Something went wrong displaying this panel
          </p>
          <p className="mt-2 text-xs text-[#9ca0b0] font-mono break-words">
            {String(this.state.error?.message || this.state.error)}
          </p>
        </div>
      );
    }

    return this.props.children;
  }
}
