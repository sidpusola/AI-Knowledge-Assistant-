import { memo, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  BookOpen, CheckCircle2, FileText, Info, Mic, PanelRightClose,
  PanelRightOpen, Paperclip, Plus, Send,
} from "lucide-react";

import DiagramPlaceholder from "./DiagramPlaceholder.jsx";
import MessageFeedback from "./MessageFeedback.jsx";
import FlowchartRenderer from "./FlowchartRenderer.jsx";
import { streamQuestion, uploadFile } from "../api.js";

/** Openers built from what is actually indexed, so a student is never
 *  invited to ask about material this install has never seen. */
function startersFor(documents) {
  if (!documents.length) return [];
  const subjects = [...new Set(documents.map((d) => d.category).filter(Boolean))];
  const first = documents[0];
  return [
    subjects[0] && `What is covered in ${subjects[0]}?`,
    first && `Summarise ${first.filename}`,
    subjects[1] && `Explain the key ideas in ${subjects[1]}`,
  ].filter(Boolean);
}

// memo matters here, not as a micro-optimisation but for correctness of
// the diagrams: typing in the composer updates ChatView state, which would
// otherwise re-render every answer, remount each MermaidDiagram and blank
// out the drawing on every keystroke.
const Message = memo(function Message({
  message,
  isStreaming = false,
  question = "",
  sessionId = null,
}) {
  // Built once per isStreaming value and reused. react-markdown keys its
  // renderers by identity, so handing it a fresh object every render makes
  // `code` look like a new component type and remounts MermaidDiagram -
  // which is what made diagrams vanish while typing.
  const markdownComponents = useMemo(
    () => ({
      // react-markdown emits <pre><code> for a fenced block. Replacing
      // only the <code> left the <pre> wrapping every diagram, which gave
      // it monospace text (a <pre> default, inherited straight into the
      // node labels) and a second padded box around the frame. A diagram
      // is not preformatted text, so the wrapper is dropped for diagram
      // fences and kept for real code.
      pre: ({ node, children, ...props }) => {
        const child = node?.children?.[0];
        const classes = child?.properties?.className || [];
        const text = child?.children?.[0]?.value || "";
        const isDiagram =
          classes.some((name) =>
            /^language-(mermaid|flowchart|flow-json)$/.test(String(name))
          ) || /"(?:diagram_)?type"\s*:\s*"flowchart"/.test(text);

        return isDiagram ? children : <pre {...props}>{children}</pre>;
      },
      // Wrap tables so a wide one scrolls sideways instead of forcing
      // display:block on the table itself, which breaks column sizing.
      table: ({ children }) => (
        <div className="chat-table-scroll">
          <table className="ans">{children}</table>
        </div>
      ),
      code: ({ className, children, ...props }) => {
        // Any of these fences may carry a diagram - ```mermaid for graph
        // syntax, ```flow-json for the card format. Which one it actually
        // is gets decided from the content, inside FlowchartRenderer.
        const source = String(children).replace(/\n$/, "");
        const fenced = /language-(mermaid|flowchart|flow-json)\b/.test(className || "");
        // A model that writes ```json instead of ```flow-json would
        // otherwise stream the whole object into the answer as ordinary
        // code. The type field is a precise enough signature to catch
        // that without hijacking a block that genuinely is teaching JSON.
        const looksLikeDiagram = /"type"\s*:\s*"flowchart"/.test(source);

        if (!fenced && !looksLikeDiagram) {
          return <code className={className} {...props}>{children}</code>;
        }
        // Nothing renderable exists until the block is complete, and the
        // half-written source is the machinery rather than the answer -
        // so a placeholder holds the space instead.
        if (isStreaming) return <DiagramPlaceholder />;
        return <FlowchartRenderer source={source} />;
      },
    }),
    [isStreaming]
  );

  if (message.role === "system") {
    return <li className="chat-system-notice"><span>{message.content}</span></li>;
  }

  if (message.role === "user") {
    return (
      <li className="msg-user">
        {/* The student's own text is shown as typed - no markdown parsing,
            so stray asterisks or underscores in a question stay literal. */}
        <div className="bubble">{message.content}</div>
      </li>
    );
  }

  // The left edge is the provenance: pine when the answer is grounded in
  // the student's own materials, grey when it came from the model's
  // general knowledge. Nothing else on the page carries that distinction,
  // and it is the thing a student most needs to know.
  const grounded = message.sources?.length > 0;

  return (
    <li className={`answer ${grounded ? "" : "general"} ${message.isError ? "is-error" : ""}`}>
      <div className="answer-flag">
        {message.isError
          ? <><Info aria-hidden="true" />Something went wrong</>
          : grounded
            ? <><CheckCircle2 aria-hidden="true" />From your course materials</>
            : <><BookOpen aria-hidden="true" />From general knowledge</>}
      </div>

      <div className="chat-markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
          {message.content}
        </ReactMarkdown>
      </div>

      {grounded && (
        <div className="sources">
          {message.sources.map((source) => (
            <span className="src" key={source}>
              <FileText aria-hidden="true" />{source}
            </span>
          ))}
        </div>
      )}

      {/* Not on an error, and not while the answer is still arriving -
          rating half an answer means nothing. */}
      {!isStreaming && !message.isError && (
        <MessageFeedback
          sessionId={sessionId}
          question={question}
          answer={message.content}
          sources={message.sources}
        />
      )}
    </li>
  );
});

function Thinking() {
  return (
    <li className="answer is-thinking">
      <div className="answer-flag"><span className="dot" /><span className="dot" /><span className="dot" /></div>
    </li>
  );
}

export default function ChatView({
  sessionId,
  documents = [],
  onDocumentsChanged,
  onNewConversation,
  asideHidden = false,
  onToggleAside,
}) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isSending, setIsSending] = useState(false);
  const [isAttaching, setIsAttaching] = useState(false);
  const fileInputRef = useRef(null);
  const scrollAnchorRef = useRef(null);

  useEffect(() => {
    scrollAnchorRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, isSending]);

  const appendMessage = (message) => {
    setMessages((prev) => [...prev, { id: crypto.randomUUID(), ...message }]);
  };

  const ask = async (text) => {
    const question = (text ?? input).trim();
    if (!question || isSending) return;

    appendMessage({ role: "user", content: question });
    setInput("");
    setIsSending(true);

    // The answer doesn't exist until the first real (non-empty) token
    // arrives - until then the thinking indicator stays visible, so a
    // stray empty chunk at the start never shows as a blank answer.
    let assistantId = null;

    const appendToAssistant = (chunk) => {
      if (assistantId === null) {
        assistantId = crypto.randomUUID();
        setMessages((prev) => [...prev, { id: assistantId, role: "assistant", content: chunk }]);
      } else {
        setMessages((prev) =>
          prev.map((m) => (m.id === assistantId ? { ...m, content: m.content + chunk } : m))
        );
      }
    };

    await streamQuestion(question, sessionId, {
      onToken: (chunk) => { if (chunk) appendToAssistant(chunk); },
      onDone: (payload) => {
        // Empty sources is not a failure - "hello" and a general-knowledge
        // answer both finish without any.
        if (assistantId !== null && payload?.sources?.length) {
          const sources = payload.sources;
          setMessages((prev) =>
            prev.map((m) => (m.id === assistantId ? { ...m, sources } : m))
          );
        }
        setIsSending(false);
      },
      onError: (message) => {
        if (assistantId === null) {
          appendMessage({
            role: "assistant",
            content: message || "Could not reach the assistant. Please try again.",
            isError: true,
          });
        } else {
          // Some of the answer already streamed in before this failed -
          // keep it, and add the error as a visible continuation rather
          // than discarding a partial but real answer.
          appendToAssistant(`\n\n${message}`);
        }
        setIsSending(false);
      },
    });
  };

  const handleAttach = async (event) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    setIsAttaching(true);
    appendMessage({ role: "system", content: `Adding “${file.name}” to your materials…` });

    try {
      const data = await uploadFile(file);
      appendMessage({ role: "system", content: data.message });
      onDocumentsChanged?.();
    } catch (error) {
      appendMessage({ role: "system", content: error.message || `Could not add “${file.name}”.` });
    } finally {
      setIsAttaching(false);
    }
  };

  const starters = startersFor(documents);
  const lastGrounded = [...messages].reverse().find((m) => m.sources?.length);

  return (
    <div className={`body with-aside ${asideHidden ? "aside-hidden" : ""}`}>
      <div className="col">
        {/* No page header any more, so the control that belongs to this
            screen lives here - top right of the column it affects, and
            clear of the sources panel beside it. */}
        <div className="col-tools">
          <button
            type="button"
            className="icon-btn hide-when-narrow"
            onClick={onToggleAside}
            aria-label={asideHidden ? "Show the sources column" : "Hide the sources column"}
            title={asideHidden ? "Show sources" : "Hide sources"}
            aria-expanded={!asideHidden}
          >
            {asideHidden
              ? <PanelRightOpen aria-hidden="true" />
              : <PanelRightClose aria-hidden="true" />}
          </button>
        </div>

        <div className="pane">
          {messages.length === 0 ? (
            <div className="opening">
              <div className="greet-unit">
                <BookOpen aria-hidden="true" />
                {documents.length
                  ? `${documents.length} ${documents.length === 1 ? "document" : "documents"} indexed`
                  : "No materials yet"}
              </div>
              <h1 className="greet">
                {documents.length
                  ? "What would you like to understand?"
                  : "Add a handout to get started."}
              </h1>
              <p className="greet-sub">
                {documents.length
                  ? "Ask anything from your materials. Every answer shows which document it came from — and says so plainly when it had to fall back on general knowledge."
                  : "Attach a PDF, Word file or set of notes with the paperclip below. Once it is indexed you can ask about it, and answers will cite the page."}
              </p>

              {starters.length > 0 && (
                <div className="starters">
                  {starters.map((text) => (
                    <button type="button" className="chip" key={text} onClick={() => ask(text)}>
                      {text}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <ul className="chat-scroll">
              {messages.map((message, index) => (
                <Message
                  key={message.id}
                  message={message}
                  sessionId={sessionId}
                  // The question this answer belongs to, so a rating
                  // records what was asked and not just what came back.
                  question={index > 0 ? messages[index - 1].content : ""}
                  // Only the final answer can still be receiving tokens.
                  isStreaming={
                    isSending && index === messages.length - 1 && message.role === "assistant"
                  }
                />
              ))}
              {isSending && <Thinking />}
              <li ref={scrollAnchorRef} aria-hidden="true" />
            </ul>
          )}
        </div>

        <div className="composer">
          <textarea
            className="composer-input"
            rows={1}
            value={input}
            placeholder={
              messages.length
                ? "Ask a follow-up — “what about the second one?” works fine…"
                : "Ask anything from your materials…"
            }
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              // Enter sends, Shift+Enter breaks the line: a student
              // pasting a multi-line question should not fire it off
              // halfway through.
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                ask();
              }
            }}
            aria-label="Ask a question"
          />
          <div className="composer-bar">
            <input
              type="file" ref={fileInputRef} onChange={handleAttach} hidden
              accept=".txt,.md,.pdf,.docx,.csv"
            />
            <button
              type="button" className="cbtn" title="Attach a document"
              aria-label="Attach a document"
              onClick={() => fileInputRef.current?.click()} disabled={isAttaching}
            >
              <Paperclip aria-hidden="true" />
            </button>
            <button
              type="button" className="cbtn" title="Voice input is not available yet"
              aria-label="Voice input is not available yet" disabled
            >
              <Mic aria-hidden="true" />
            </button>

            {messages.length > 0 && (
              <button type="button" className="chip" onClick={onNewConversation}>
                <Plus aria-hidden="true" />New conversation
              </button>
            )}

            <div className="grow" />
            <button
              type="button" className="send"
              onClick={() => ask()} disabled={isSending || !input.trim()}
            >
              Send <Send aria-hidden="true" />
            </button>
          </div>
        </div>
      </div>

      <div className="aside">
        <h4 className="aside-h">Sources for the latest answer</h4>
        {lastGrounded ? (
          lastGrounded.sources.map((source) => (
            <div className="src-row" key={source}>
              <div className="src-name"><FileText aria-hidden="true" />{source}</div>
            </div>
          ))
        ) : (
          <p className="src-quote" style={{ marginTop: 0 }}>
            Citations appear here once an answer is grounded in one of your
            documents.
          </p>
        )}

      </div>
    </div>
  );
}
