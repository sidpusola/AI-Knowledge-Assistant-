import { useState } from "react";
import { ThumbsDown, ThumbsUp } from "lucide-react";

import { sendFeedback } from "../api.js";

/**
 * Thumbs up/down under an answer.
 *
 * This is the only signal in the system that comes from a student
 * rather than a metric. Retrieval accuracy and latency say whether the
 * machinery works; this says whether the answer actually helped, which
 * nothing else measures.
 *
 * The choice is recorded locally the moment it is clicked and the
 * request is fire-and-forget. A rating is diagnostic, not part of the
 * product: making someone wait on it, or showing them an error for
 * having tried to help, would be worse than quietly losing the odd one.
 */
export default function MessageFeedback({ sessionId, question, answer, sources }) {
  const [rating, setRating] = useState(null);

  const choose = (value) => {
    // Clicking the same thumb again clears it - a misclick should be
    // undoable, and an unrated answer is more honest than a wrong rating.
    const next = rating === value ? null : value;
    setRating(next);
    if (next) {
      sendFeedback({ sessionId, rating: next, question, answer, sources });
    }
  };

  return (
    <div className="msg-feedback">
      <span className="msg-feedback-label">
        {rating ? "Thanks" : "Was this helpful?"}
      </span>
      <button
        type="button"
        className={`msg-feedback-button ${rating === "up" ? "is-chosen" : ""}`}
        onClick={() => choose("up")}
        aria-pressed={rating === "up"}
        aria-label="This answer was helpful"
        title="Helpful"
      >
        <ThumbsUp size={13} strokeWidth={2} />
      </button>
      <button
        type="button"
        className={`msg-feedback-button ${rating === "down" ? "is-chosen" : ""}`}
        onClick={() => choose("down")}
        aria-pressed={rating === "down"}
        aria-label="This answer was not helpful"
        title="Not helpful"
      >
        <ThumbsDown size={13} strokeWidth={2} />
      </button>
    </div>
  );
}
