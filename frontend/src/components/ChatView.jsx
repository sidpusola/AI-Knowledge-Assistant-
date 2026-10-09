import { memo, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import DiagramPlaceholder from "./DiagramPlaceholder.jsx";
import MessageFeedback from "./MessageFeedback.jsx";
import FlowchartRenderer from "./FlowchartRenderer.jsx";
import { streamQuestion, uploadFile } from "../api.js";

const WELCOME_MESSAGE = {
  id: "welcome",
  role: "assistant",
  content:
    "Hi! I'm your learning assistant. Ask me anything about your course materials — or attach a document with the paperclip below and I'll learn from it too.",
};

// memo matters here, not as a micro-optimisation but for correctness of
// the diagrams: typing in the composer updates ChatView state, which would
// otherwise re-render every bubble, remount each MermaidDiagram and blank
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
      // only the <code> left the <pre> wrapping every diagram, which
      // gave it monospace text (a <pre> default, inherited straight
      // into the node labels) and a second padded box around the
      // frame. A diagram is not preformatted text, so the wrapper is
      // dropped for diagram fences and kept for real code.
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
      // Wrap tables so a wide one scrolls sideways inside the bubble,
      // instead of forcing display:block on the table itself (which
      // breaks normal column sizing).
      table: ({ children }) => (
        <div className="chat-table-scroll">
          <table>{children}</table>
        </div>
      ),
      // A ```mermaid block becomes a drawn diagram instead of source
      // text. Deliberately NOT drawn while the answer is still
      // streaming: a diagram arrives a few characters at a time and
      // every intermediate state is invalid Mermaid, so rendering
      // mid-stream means a burst of parse failures and a flickering
      // bubble. The source shows until the stream finishes, then it
      // resolves into the picture.
      code: ({ className, children, ...props }) => {
        // Any of these fences may carry a diagram - ```mermaid for graph
        // syntax, ```flow-json for the card format. Which one it actually
        // is gets decided from the content, inside FlowchartRenderer.
        // Everything else is ordinary code and renders as it always did.
        const source = String(children).replace(/\n$/, "");
        const fenced = /language-(mermaid|flowchart|flow-json)\b/.test(className || "");
        // A model that writes ```json instead of ```flow-json would
        // otherwise stream the whole object into the bubble as ordinary
        // code. The type field is a precise enough signature to catch
        // that without hijacking a block that genuinely is teaching JSON.
        const looksLikeDiagram = /"type"\s*:\s*"flowchart"/.test(source);

        if (!fenced && !looksLikeDiagram) {
          return (
            <code className={className} {...props}>
              {children}
            </code>
          );
        }
        // Nothing renderable exists until the block is complete, and the
        // half-written source is the machinery rather than the answer -
        // so a placeholder holds the space instead. See DiagramPlaceholder.
        if (isStreaming) {
          return <DiagramPlaceholder />;
        }
        return <FlowchartRenderer source={source} />;
      },
    }),
    [isStreaming]
  );

  if (message.role === "system") {
    return (
      <li className="chat-system-notice">
        <span>{message.content}</span>
      </li>
    );
  }

  const isUser = message.role === "user";
  // A flowchart needs width in a way a paragraph does not, so a
  // bubble carrying one is allowed to run wider than the 75% that
  // keeps prose readable.
  const hasDiagram = !isUser && /```(mermaid|flowchart|flow-json)/.test(message.content || "");
  return (
    <li className={`chat-row ${isUser ? "chat-row-user" : "chat-row-assistant"}`}>
      <div className={`chat-bubble ${isUser ? "chat-bubble-user" : "chat-bubble-assistant"} ${message.isError ? "chat-bubble-error" : ""} ${hasDiagram ? "chat-bubble-wide" : ""}`}>
        {isUser ? (
          // The student's own text is shown as typed - no markdown parsing,
          // so stray asterisks or underscores in a question stay literal.
          message.content
        ) : (
          <div className="chat-markdown">
            {/* remark-gfm is what makes tables work - they're a GitHub
                Flavored Markdown extension, not core Markdown, so without
                this plugin a table renders as a run-on paragraph of pipes. */}
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={markdownComponents}
            >
              {message.content}
            </ReactMarkdown>

            {/* Not on the welcome message, not on an error, and not
                while the answer is still arriving - rating half an
                answer means nothing. */}
            {!isStreaming && !message.isError && message.id !== "welcome" && (
              <MessageFeedback
                sessionId={sessionId}
                question={question}
                answer={message.content}
                sources={message.sources}
              />
            )}

            {message.sources?.length > 0 && (
              <div className="chat-sources">
                <span className="chat-sources-label">Sources</span>
                {message.sources.map((source) => (
                  <span key={source} className="chat-source-pill">
                    {source}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </li>
  );
});

function TypingIndicator() {
  return (
    <li className="chat-row chat-row-assistant">
      <div className="chat-bubble chat-bubble-assistant chat-typing">
        <span className="chat-typing-dot" />
        <span className="chat-typing-dot" />
        <span className="chat-typing-dot" />
      </div>
    </li>
  );
}

export default function ChatView({ sessionId, onDocumentsChanged }) {
  const [messages, setMessages] = useState([WELCOME_MESSAGE]);
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

  const handleSend = async () => {
    const question = input.trim();
    if (!question || isSending) return;

    appendMessage({ role: "user", content: question });
    setInput("");
    setIsSending(true);

    // The assistant bubble doesn't exist until the first real (non-empty)
    // token arrives - until then the typing indicator stays visible, so a
    // stray empty chunk at the start of a stream never shows as a blank bubble.
    let assistantId = null;

    const appendToAssistant = (text) => {
      if (assistantId === null) {
        assistantId = crypto.randomUUID();
        setMessages((prev) => [...prev, { id: assistantId, role: "assistant", content: text }]);
      } else {
        setMessages((prev) =>
          prev.map((message) =>
            message.id === assistantId ? { ...message, content: message.content + text } : message
          )
        );
      }
    };

    await streamQuestion(question, sessionId, {
      onToken: (chunk) => {
        if (chunk) appendToAssistant(chunk);
      },
      onDone: (payload) => {
        // Empty materials, not a stream failure - "hello" or a
        // general-knowledge answer both finish with no sources.
        if (assistantId !== null && payload?.sources?.length) {
          const sources = payload.sources;
          setMessages((prev) =>
            prev.map((message) => (message.id === assistantId ? { ...message, sources } : message))
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
          // than discarding a partial (but real) answer.
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
    appendMessage({ role: "system", content: `Adding "${file.name}" to the knowledge base...` });

    try {
      const data = await uploadFile(file);
      appendMessage({ role: "system", content: `✓ ${data.message}` });
      onDocumentsChanged?.();
    } catch (error) {
      appendMessage({
        role: "system",
        content: error.message || `Could not add "${file.name}".`,
      });
    } finally {
      setIsAttaching(false);
    }
  };

  return (
    <div className="chat-view">
      <ul className="chat-log">
        {messages.map((message, index) => (
          <Message
            key={message.id}
            message={message}
            sessionId={sessionId}
            // The question this answer belongs to, so a rating records
            // what was asked and not just what came back.
            question={index > 0 ? messages[index - 1].content : ""}
            // Only the final assistant bubble can still be receiving tokens.
            isStreaming={
              isSending && index === messages.length - 1 && message.role === "assistant"
            }
          />
        ))}
        {isSending && <TypingIndicator />}
        <li ref={scrollAnchorRef} aria-hidden="true" />
      </ul>

      <div className="chat-input-bar">
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleAttach}
          hidden
          accept=".txt,.md,.pdf,.docx,.csv"
        />
        <button
          type="button"
          className="chat-attach-button"
          title="Attach a document"
          onClick={() => fileInputRef.current?.click()}
          disabled={isAttaching}
        >
          📎
        </button>

        <input
          type="text"
          className="chat-text-input"
          placeholder="Ask a question about your course materials..."
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") handleSend();
          }}
        />

        <button
          type="button"
          className="chat-send-button"
          onClick={handleSend}
          disabled={isSending || !input.trim()}
        >
          Send
        </button>
      </div>
    </div>
  );
}
