const API_BASE = "http://127.0.0.1:8000";

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

export async function listDocuments() {
  const response = await fetch(`${API_BASE}/documents`);
  return parseJsonOrThrow(response);
}

export async function resetSession(sessionId) {
  if (!sessionId) return;
  await fetch(`${API_BASE}/sessions/${sessionId}`, { method: "DELETE" }).catch(() => {
    // best-effort - a failed cleanup call shouldn't block starting a new conversation
  });
}
