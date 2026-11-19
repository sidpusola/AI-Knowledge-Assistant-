import { useState } from "react";
import { BookOpen, Briefcase, Check, CircleDashed, Lightbulb, Mic, X } from "lucide-react";

import { answerInterview, startInterview } from "../api.js";
import { initialDoc } from "../lib/study.js";
import { DocPicker, ErrorNote, NoDocuments, SignInNote, Working } from "./StudyCommon.jsx";

const SCORE_WORDS = ["", "Needs work", "Getting there", "Solid start", "Strong", "Excellent"];

// The score is computed from these, so they are shown: a student can see
// exactly which points cost them.
const COVERAGE = {
  covered: { Icon: Check, word: "Covered" },
  partly: { Icon: CircleDashed, word: "Partly" },
  missing: { Icon: X, word: "Missed" },
};

export default function PlacementView({ documents, focusDoc, user, onNavigate }) {
  const [docId, setDocId] = useState(() => initialDoc(documents, focusDoc));
  const [question, setQuestion] = useState(null);   // { interview_id, question, source }
  const [answer, setAnswer] = useState("");
  const [feedback, setFeedback] = useState(null);
  const [busy, setBusy] = useState(null);           // "question" | "feedback"
  const [error, setError] = useState(null);

  if (!documents.length) return <Shell><NoDocuments onNavigate={onNavigate} /></Shell>;

  const chosenDoc = docId || initialDoc(documents, focusDoc);

  const ask = async () => {
    setBusy("question");
    setError(null);
    setQuestion(null);
    setFeedback(null);
    setAnswer("");
    try {
      setQuestion(await startInterview(chosenDoc));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(null);
    }
  };

  const submit = async (event) => {
    event.preventDefault();
    if (!answer.trim()) return;
    setBusy("feedback");
    setError(null);
    try {
      setFeedback(await answerInterview(question.interview_id, answer));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <Shell>
      <div className="study-bar">
        <DocPicker
          documents={documents}
          value={chosenDoc}
          onChange={(value) => { setDocId(value); setQuestion(null); setFeedback(null); }}
          disabled={Boolean(busy)}
        />
        <button type="button" className="btn btn-primary" onClick={ask} disabled={!chosenDoc || Boolean(busy)}>
          {question ? "Next question" : "Start practice"}
        </button>
      </div>

      <SignInNote user={user} what="your interview scores" />
      <ErrorNote message={error} />

      {busy === "question" && <Working label="Thinking of an interview question…" hint="A few seconds." />}

      {!question && !busy && (
        <div className="card pad study-intro">
          <p>
            Interviewers ask about what your course taught you. Pick a document,
            get a question an interviewer could ask about it, and answer the
            way you would out loud. You'll get a rating, what went well, what
            to add, and an example of a strong answer.
          </p>
        </div>
      )}

      {question && (
        <div className="card pad iv-card">
          <div className="iv-label"><Briefcase aria-hidden="true" />Interviewer asks</div>
          <h2 className="iv-q">“{question.question}”</h2>

          {!feedback && (
            <form onSubmit={submit}>
              <label className="iv-answer">
                <span className="visually-hidden">Your answer</span>
                <textarea
                  value={answer}
                  onChange={(event) => setAnswer(event.target.value)}
                  placeholder="Answer as you would in the interview — a few clear sentences."
                  rows={6}
                  maxLength={4000}
                  disabled={busy === "feedback"}
                />
              </label>
              <div className="iv-actions">
                <span className="iv-count">{answer.trim().split(/\s+/).filter(Boolean).length} words</span>
                <button type="submit" className="btn btn-primary" disabled={!answer.trim() || busy === "feedback"}>
                  <Mic aria-hidden="true" />{busy === "feedback" ? "Scoring…" : "Get feedback"}
                </button>
              </div>
            </form>
          )}
        </div>
      )}

      {feedback && <Feedback feedback={feedback} answer={answer} user={user} />}
    </Shell>
  );
}

function Feedback({ feedback, answer, user }) {
  return (
    <div className="iv-feedback">
      <div className="card pad iv-score">
        <div className="iv-meter" role="img" aria-label={`Rated ${feedback.score} out of 5`}>
          {[1, 2, 3, 4, 5].map((n) => (
            <span key={n} className={n <= feedback.score ? "on" : ""} />
          ))}
        </div>
        <div>
          <div className="iv-score-n">{feedback.score}/5 · {SCORE_WORDS[feedback.score]}</div>
          {user && <div className="iv-score-s">Saved to your progress</div>}
        </div>
      </div>

      <div className="iv-grid">
        {feedback.strengths.length > 0 && (
          <section className="card pad">
            <h3 className="guide-h3"><Check aria-hidden="true" />What went well</h3>
            <ul className="guide-points">
              {feedback.strengths.map((s) => <li key={s}>{s}</li>)}
            </ul>
          </section>
        )}
        {feedback.improvements.length > 0 && (
          <section className="card pad">
            <h3 className="guide-h3"><Lightbulb aria-hidden="true" />To make it stronger</h3>
            <ul className="guide-points">
              {feedback.improvements.map((s) => <li key={s}>{s}</li>)}
            </ul>
          </section>
        )}
      </div>

      <section className="card pad">
        <h3 className="guide-h3">Your answer</h3>
        <p className="iv-yours">{answer}</p>
        {feedback.model_answer && (
          <>
            <h3 className="guide-h3">A strong answer</h3>
            <p className="iv-model">{feedback.model_answer}</p>
          </>
        )}
        <h3 className="guide-h3">Points to cover</h3>
        <ul className="iv-points">
          {feedback.key_points.map((point, i) => {
            const verdict = feedback.coverage?.[i];
            const { Icon, word } = COVERAGE[verdict] || {};
            return (
              <li key={point} className={verdict || ""}>
                {Icon && (
                  <span className="iv-verdict"><Icon aria-hidden="true" />{word}</span>
                )}
                <span>{point}</span>
              </li>
            );
          })}
        </ul>
        <p className="guide-sources"><BookOpen aria-hidden="true" />From your material: {feedback.source}</p>
      </section>
    </div>
  );
}

function Shell({ children }) {
  return (
    <div className="body">
      <div className="pane study">
        <h1 className="page">Placement prep</h1>
        <p className="page-sub">Practise answering interview questions about what your course taught you.</p>
        {children}
      </div>
    </div>
  );
}
