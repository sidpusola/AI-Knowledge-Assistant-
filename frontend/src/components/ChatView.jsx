import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { streamQuestion, uploadFile } from "../api.js";

const WELCOME_MESSAGE = {
  id: "welcome",
  role: "assistant",
  content:
    "Hi! I'm your learning assistant. Ask me anything about your course materials — or attach a document with the paperclip below and I'll learn from it too.",
};

function Message({ message }) {
  if (message.role === "system") {
    return (
      <li className="chat-system-notice">
        <span>{message.content}</span>
      </li>
    );
  }

  const isUser = message.role === "user";
  return (
    <li className={`chat-row ${isUser ? "chat-row-user" : "chat-row-assistant"}`}>
      <div className={`chat-bubble ${isUser ? "chat-bubble-user" : "chat-bubble-assistant"} ${message.isError ? "chat-bubble-error" : ""}`}>
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
              components={{
                // Wrap tables so a wide one scrolls sideways inside the
                // bubble, instead of forcing display:block on the table
                // itself (which breaks normal column sizing).
                table: ({ children }) => (
                  <div className="chat-table-scroll">
                    <table>{children}</table>
                  </div>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </li>
  );
}

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
      onDone: () => {
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
        {messages.map((message) => (
          <Message key={message.id} message={message} />
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
