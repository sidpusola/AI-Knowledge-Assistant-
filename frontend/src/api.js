/* In development the Vite dev server proxies /api to the backend (see
 * vite.config.js), so every request is same-origin: no CORS preflight,
 * and the session cookie is an ordinary first-party cookie rather than
 * a cross-site one that SameSite=Lax withholds.
 *
 * A build served from somewhere other than the API needs the absolute
 * address, which VITE_API_BASE supplies. */
const BACKEND_ORIGIN = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";
const API_BASE = import.meta.env.DEV ? "/api" : BACKEND_ORIGIN;

/** The host the session cookie belongs to. Cookies are scoped by host
 *  and ignore the port, so a page on a different hostname - localhost
 *  versus 127.0.0.1 - can never send it, however the request is made. */
export const COOKIE_HOST = new URL(BACKEND_ORIGIN).hostname;

async function parseJsonOrThrow(response) {
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(data?.detail || "Something went wrong. Please try again.");
  }
  return data;
}

export async function askQuestion(question, sessionId) {
  const response = await fetch(`${API_BASE}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id: sessionId }),
  });
  return parseJsonOrThrow(response);
}

// Streams an answer via Server-Sent Events. Uses fetch + a manual SSE
// parser rather than EventSource, since EventSource can't send a POST
// body (we need to send the question + session_id).
//
// callbacks:
//   onToken(text)     - called for each chunk of the answer as it arrives
//   onDone({session_id}) - called once, after the full answer has streamed
//   onError(message)  - called on a failure, whether before the first
//                       chunk (a normal HTTP error) or mid-stream (an
//                       "error" SSE event - the status is already 200
//                       by then, so it can't be reported as an HTTP code)
export async function streamQuestion(question, sessionId, { onToken, onDone, onError } = {}) {
  let response;
  try {
    response = await fetch(`${API_BASE}/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId }),
    });
  } catch {
    onError?.("Could not reach the assistant. Please try again.");
    return;
  }

  if (!response.ok || !response.body) {
    const data = await response.json().catch(() => null);
    onError?.(data?.detail || "Could not reach the assistant. Please try again.");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const handleEvent = (rawEvent) => {
    const lines = rawEvent.split("\n");
    const eventType = lines.find((line) => line.startsWith("event:"))?.slice(6).trim() || "message";
    const dataLine = lines.find((line) => line.startsWith("data:"))?.slice(5).trim();
    if (dataLine === undefined) return;

    let payload;
    try {
      payload = JSON.parse(dataLine);
    } catch {
      return;
    }

    if (eventType === "token") onToken?.(payload);
    else if (eventType === "done") onDone?.(payload);
    else if (eventType === "error") onError?.(payload);
  };

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // SSE events are separated by a blank line; keep any trailing
    // incomplete event in the buffer for the next chunk.
    const events = buffer.split("\n\n");
    buffer = events.pop() ?? "";

    for (const rawEvent of events) {
      if (rawEvent.trim()) handleEvent(rawEvent);
    }
  }
}

export async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE}/upload`, {
    method: "POST",
    body: formData,
  });
  return parseJsonOrThrow(response);
}

/** Index a web page. The backend has had /ingest-url since Week 3; the
 *  UI never offered a way to reach it. */
export async function ingestUrl(url) {
  const response = await fetch(`${API_BASE}/ingest-url`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  return parseJsonOrThrow(response);
}

export async function listDocuments() {
  const response = await fetch(`${API_BASE}/documents`);
  return parseJsonOrThrow(response);
}

export async function deleteDocument(docId) {
  const response = await fetch(`${API_BASE}/documents/${docId}`, { method: "DELETE" });
  return parseJsonOrThrow(response);
}

export async function resetSession(sessionId) {
  if (!sessionId) return;
  await fetch(`${API_BASE}/sessions/${sessionId}`, { method: "DELETE" }).catch(() => {
    // best-effort - a failed cleanup call shouldn't block starting a new conversation
  });
}

/** Record a thumbs up/down on an answer.
 *
 * Deliberately swallows its own failures. A rating is diagnostic, not
 * part of the product: if the request fails the student should not see
 * an error for having tried to help, and the UI has already shown their
 * choice. The server treats it the same way - see feedback.py.
 */
export async function sendFeedback({ sessionId, rating, question, answer, sources }) {
  try {
    await fetch(`${API_BASE}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        rating,
        question: (question || "").slice(0, 1000),
        // Enough to recognise the answer later without storing an essay
        // per vote; the server caps this too.
        answer_excerpt: (answer || "").slice(0, 2000),
        sources: sources || [],
      }),
    });
  } catch {
    // nothing useful to tell the student mid-conversation
  }
}

/* --- accounts -------------------------------------------------------
 *
 * These all pass credentials: "include". The session is an HttpOnly
 * cookie, and fetch does not send cookies cross-origin by default - the
 * page is served from :5173 and the API from :8000, so without this the
 * browser silently omits it and every call looks signed-out.
 */

/** Where to send the browser to start Google sign-in. A full page
 *  navigation, not a fetch: the flow is a series of redirects through
 *  Google and ends by setting a cookie. */
export function googleLoginUrl() {
  // Straight to the backend, not through the dev proxy. Two reasons:
  // the server bounces this request to the host its OAuth redirect URI
  // names, and behind the proxy it would see the dev server's address
  // and bounce to a path that does not exist. And the cookies the flow
  // sets have to belong to the backend's host anyway - which is the
  // same host as the dev server, so the page still gets them.
  return `${BACKEND_ORIGIN}/auth/google/login`;
}

/** The signed-in user, or null. Never throws - called on first paint,
 *  and a backend that isn't up yet should show the signed-out state
 *  rather than a blank page. */
export async function fetchMe() {
  try {
    const response = await fetch(`${API_BASE}/auth/me`, { credentials: "include" });
    if (!response.ok) return null;
    return (await response.json()).user;
  } catch {
    return null;
  }
}

/** Whether Google sign-in is set up on the server, so the UI can say
 *  "not configured" instead of offering a button that 503s. */
export async function fetchAuthStatus() {
  try {
    const response = await fetch(`${API_BASE}/auth/status`, { credentials: "include" });
    if (!response.ok) return { configured: false };
    return await response.json();
  } catch {
    return { configured: false };
  }
}

export async function saveProfile({ displayName, dateOfBirth }) {
  const response = await fetch(`${API_BASE}/auth/profile`, {
    method: "POST",
    credentials: "include",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      display_name: displayName,
      date_of_birth: dateOfBirth || null,
    }),
  });
  return (await parseJsonOrThrow(response)).user;
}

export async function logout() {
  try {
    await fetch(`${API_BASE}/auth/logout`, { method: "POST", credentials: "include" });
  } catch {
    // The cookie is cleared by the server; if the call fails the local
    // state is reset anyway and the next request will be unauthorised.
  }
}
