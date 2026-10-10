import { useCallback, useEffect, useState } from "react";

import "./styles/fonts.css";
import "./styles/tokens.css";
import "./styles/shell.css";
import "./App.css";
import "./styles/screens.css";

import AppShell from "./components/AppShell.jsx";
import ChatView from "./components/ChatView.jsx";
import DocumentsView from "./components/DocumentsView.jsx";
import {
  COOKIE_HOST, fetchAuthStatus, fetchMe, listDocuments, resetSession,
} from "./api.js";

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

/** Read ?signed_in / ?auth_error, then scrub them from the address bar.
 *  Leaving them there means a refresh re-shows a stale banner, and the
 *  URL a student copies carries a sign-in artefact. */
function takeAuthOutcome() {
  const params = new URLSearchParams(window.location.search);
  const error = params.get("auth_error");

  if (error || params.get("signed_in")) {
    params.delete("auth_error");
    params.delete("signed_in");
    const query = params.toString();
    window.history.replaceState(
      {}, "", window.location.pathname + (query ? `?${query}` : "")
    );
  }
  return error;
}

/** Signing in sets a cookie on the backend's host. Served from any
 *  other hostname, the browser will never send it back and the app sits
 *  there looking permanently signed-out with nothing to explain why -
 *  which is exactly how this was found. Better to say it out loud. */
function wrongHostForCookies() {
  const here = window.location.hostname;
  if (here === COOKIE_HOST) return null;
  return `${window.location.protocol}//${COOKIE_HOST}:${window.location.port}`;
}

/** Panel choices are remembered. Collapsing the rail is a working
 *  preference, not a one-off - having to redo it on every reload is
 *  the kind of small friction that makes people stop using the toggle. */
function stored(key, fallback = false) {
  try {
    const saved = localStorage.getItem(key);
    return saved === null ? fallback : saved === "1";
  } catch {
    return fallback;
  }
}

function App() {
  const [sessionId, setSessionId] = useState(getOrCreateSessionId);
  const [activeTab, setActiveTab] = useState("chat");
  const [theme, setTheme] = useState(getInitialTheme);
  const [user, setUser] = useState(null);
  const [authConfigured, setAuthConfigured] = useState(false);
  const [authError, setAuthError] = useState(takeAuthOutcome);
  const [correctOrigin] = useState(wrongHostForCookies);
  const [railCollapsed, setRailCollapsed] = useState(() => stored("aika_rail"));
  const [asideHidden, setAsideHidden] = useState(() => stored("aika_aside"));
  const [documents, setDocuments] = useState([]);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    localStorage.setItem("aika_theme", theme);
  }, [theme]);

  useEffect(() => {
    localStorage.setItem("aika_rail", railCollapsed ? "1" : "0");
  }, [railCollapsed]);

  useEffect(() => {
    localStorage.setItem("aika_aside", asideHidden ? "1" : "0");
  }, [asideHidden]);

  // Who is signed in, asked once on load. The cookie is HttpOnly, so
  // the page cannot read it - only the server can say.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const [me, status] = await Promise.all([fetchMe(), fetchAuthStatus()]);
      if (cancelled) return;
      setUser(me);
      setAuthConfigured(Boolean(status?.configured));
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const refreshDocuments = useCallback(async () => {
    try {
      const data = await listDocuments();
      setDocuments(data?.documents || []);
    } catch {
      // The documents list is context, not the product. A failure here
      // should not stop someone asking a question.
    }
  }, []);

  useEffect(() => {
    // Wrapped rather than called directly: the React Compiler reads a
    // bare call whose body sets state as a synchronous cascade, even
    // when the state change only happens after an await.
    (async () => {
      await refreshDocuments();
    })();
  }, [refreshDocuments]);

  const startNewConversation = () => {
    resetSession(sessionId);
    const next = crypto.randomUUID();
    localStorage.setItem("aika_session_id", next);
    setSessionId(next);
  };

  return (
    <AppShell
      activeTab={activeTab}
      onTabChange={setActiveTab}
      theme={theme}
      onToggleTheme={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
      user={user}
      authConfigured={authConfigured}
      onUserChange={setUser}
      railCollapsed={railCollapsed}
      onToggleRail={() => setRailCollapsed((v) => !v)}
    >
      {correctOrigin && (
        <div className="app-auth-error" role="alert">
          <span>
            Signing in will not stick on this address. Open{" "}
            <a href={correctOrigin}>{correctOrigin}</a> instead — the session
            cookie belongs to <code>{COOKIE_HOST}</code> and a browser will not
            send it from a different hostname.
          </span>
        </div>
      )}

      {authError && (
        <div className="app-auth-error" role="alert">
          <span>{authError}</span>
          <button type="button" onClick={() => setAuthError(null)} aria-label="Dismiss">
            ×
          </button>
        </div>
      )}

      {activeTab === "chat" ? (
        <ChatView
          key={sessionId}
          sessionId={sessionId}
          documents={documents}
          onDocumentsChanged={refreshDocuments}
          onNewConversation={startNewConversation}
          asideHidden={asideHidden}
          onToggleAside={() => setAsideHidden((v) => !v)}
        />
      ) : (
        <DocumentsView
          documents={documents}
          onDocumentsChanged={refreshDocuments}
        />
      )}
    </AppShell>
  );
}

export default App;
