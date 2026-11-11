// src/test-setup.js
// Registers @testing-library/jest-dom's matchers (toBeInTheDocument,
// toHaveClass, etc.) globally for every vitest file — previously absent,
// so tests could only assert via raw DOM/className checks (see
// ControlBar.test.jsx for what that looked like before this existed).
// Wired in via vite.config.js's test.setupFiles.
import "@testing-library/jest-dom/vitest";
