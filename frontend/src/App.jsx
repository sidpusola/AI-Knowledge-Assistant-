import { useEffect, useState } from "react";
import "./App.css";
import ChatView from "./components/ChatView.jsx";
import KnowledgeBase from "./components/KnowledgeBase.jsx";
import { resetSession } from "./api.js";

function getOrCreateSessionId() {
  const existing = localStorage.getItem("aika_session_id");
  if (existing) return existing;

  const created = crypto.randomUUID();
  localStorage.setItem("aika_session_id", created);
  return created;
}

function getInitialTheme() {
  const saved = localStorage.getItem("aika_theme");
  if (saved === "light" || saved === "dark") return saved;

  // No explicit choice yet - follow whatever the device is set to.
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function App() {
  const [sessionId, setSessionId] = useState(getOrCreateSessionId);
  const [activeTab, setActiveTab] = useState("chat");
  const [kbRefreshKey, setKbRefreshKey] = useState(0);
  const [theme, setTheme] = useState(getInitialTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("aika_theme", theme);
  }, [theme]);

  const startNewConversation = () => {
    resetSession(sessionId);
    const next = crypto.randomUUID();
    localStorage.setItem("aika_session_id", next);
    setSessionId(next);
  };

  const bumpKbRefresh = () => setKbRefreshKey((key) => key + 1);
  const toggleTheme = () => setTheme((current) => (current === "dark" ? "light" : "dark"));

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-brand">
          <span className="app-brand-mark">🎓</span>
          <div>
            <h1>Learning Assistant</h1>
            <p className="app-tagline">Ask questions, get grounded answers from your course materials</p>
          </div>
        </div>

        <nav className="app-tabs">
          <button
            type="button"
            className={activeTab === "chat" ? "app-tab app-tab-active" : "app-tab"}
            onClick={() => setActiveTab("chat")}
          >
            Chat
          </button>
          <button
            type="button"
            className={activeTab === "kb" ? "app-tab app-tab-active" : "app-tab"}
            onClick={() => setActiveTab("kb")}
          >
            Knowledge base
          </button>
        </nav>

        <div className="app-header-actions">
          <button
            type="button"
            className="app-theme-toggle"
            onClick={toggleTheme}
            title={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
            aria-label={theme === "dark" ? "Switch to light mode" : "Switch to dark mode"}
          >
            {theme === "dark" ? "☀" : "☾"}
          </button>

          {activeTab === "chat" && (
            <button type="button" className="app-new-chat" onClick={startNewConversation}>
              + New conversation
            </button>
          )}
        </div>
      </header>

      <main className="app-main">
        {activeTab === "chat" ? (
          <ChatView key={sessionId} sessionId={sessionId} onDocumentsChanged={bumpKbRefresh} />
        ) : (
          <KnowledgeBase refreshKey={kbRefreshKey} onDocumentsChanged={bumpKbRefresh} />
        )}
      </main>
    </div>
  );
}

export default App;
