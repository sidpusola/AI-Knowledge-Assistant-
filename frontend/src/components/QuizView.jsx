import { useState } from "react";
import { BookOpen, Check, RotateCcw, X } from "lucide-react";

import { answerQuiz, startQuiz } from "../api.js";
import { initialDoc } from "../lib/study.js";
import { DocPicker, ErrorNote, NoDocuments, SignInNote, Working } from "./StudyCommon.jsx";

const LETTERS = ["A", "B", "C", "D"];

export default function QuizView({ documents, focusDoc, user, onNavigate }) {
  const [docId, setDocId] = useState(() => initialDoc(documents, focusDoc));
  const [quiz, setQuiz] = useState(null);          // { quiz_id, filename, questions }
  const [index, setIndex] = useState(0);
  const [result, setResult] = useState(null);      // the server's verdict on this question
  const [showScore, setShowScore] = useState(false);
  const [busy, setBusy] = useState(null);          // "start" | "answer"
  const [error, setError] = useState(null);

  if (!documents.length) return <Shell><NoDocuments onNavigate={onNavigate} /></Shell>;

  const chosenDoc = docId || initialDoc(documents, focusDoc);

  const begin = async () => {
    setBusy("start");
    setError(null);
    setQuiz(null);
    setResult(null);
    setShowScore(false);
    setIndex(0);
    try {
      setQuiz(await startQuiz(chosenDoc, 5));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(null);
    }
  };

  const choose = async (choice) => {
    if (result || busy) return;
    setBusy("answer");
    setError(null);
    try {
      setResult(await answerQuiz(quiz.quiz_id, index, choice));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(null);
    }
  };

  const next = () => {
    if (index === quiz.questions.length - 1) {
      setShowScore(true);
      return;
    }
    setResult(null);
    setIndex((i) => i + 1);
  };

  const question = quiz?.questions[index];
  const finished = showScore && result;

  return (
    <Shell>
      <div className="study-bar">
        <DocPicker
          documents={documents}
          value={chosenDoc}
          onChange={(value) => { setDocId(value); setQuiz(null); setResult(null); setShowScore(false); }}
          disabled={busy === "start"}
        />
        <button type="button" className="btn btn-primary" onClick={begin} disabled={!chosenDoc || busy === "start"}>
          {quiz ? "New quiz" : "Start quiz"}
        </button>
      </div>

      <SignInNote user={user} what="quiz scores" />
      <ErrorNote message={error} />

      {busy === "start" && (
        <Working
          label="Writing five questions from this document…"
          hint="Each question comes from a different passage. This usually takes 20–40 seconds."
        />
      )}

      {!quiz && !busy && (
        <div className="card pad study-intro">
          <p>
            Five multiple-choice questions drawn from across the document. Every
            question comes from a passage in your own material, and after each
            answer you'll see why — and which page to read again.
          </p>
        </div>
      )}

      {question && !finished && (
        <div className="card pad quiz-card">
          <div className="quiz-meta">
            Question {index + 1} of {quiz.questions.length}
            <span className="quiz-doc">{quiz.filename}</span>
          </div>
          <div className="quiz-progress" aria-hidden="true">
            <span style={{ width: `${((index + (result ? 1 : 0)) / quiz.questions.length) * 100}%` }} />
          </div>
          <h2 className="quiz-q">{question.question}</h2>

          <div className="quiz-options" role="group" aria-label="Answer options">
            {question.options.map((option, i) => {
              let state = "";
              if (result) {
                if (i === result.answer) state = "is-right";
                else if (i === result.chosen) state = "is-wrong";
                else state = "is-dim";
              }
              return (
                <button
                  key={i}
                  type="button"
                  className={`quiz-opt ${state}`}
                  onClick={() => choose(i)}
                  disabled={Boolean(result) || busy === "answer"}
                >
                  <span className="quiz-letter">{LETTERS[i]}</span>
                  <span className="quiz-text">{option}</span>
                  {state === "is-right" && <Check aria-label="Correct answer" />}
                  {state === "is-wrong" && <X aria-label="Your answer" />}
                </button>
              );
            })}
          </div>

          {result && (
            <div className={`quiz-verdict ${result.correct ? "right" : "wrong"}`} role="status">
              <strong>{result.correct ? "Correct." : "Not quite."}</strong> {result.explanation}
              <div className="quiz-src"><BookOpen aria-hidden="true" />Read again: {result.source}</div>
            </div>
          )}

          {result && (
            <div className="quiz-actions">
              <button type="button" className="btn btn-primary" onClick={next}>
                {index === quiz.questions.length - 1 ? "See score" : "Next question"}
              </button>
            </div>
          )}
        </div>
      )}

      {finished && <ScoreCard result={result} filename={quiz.filename} onAgain={begin} user={user} />}
    </Shell>
  );
}

function ScoreCard({ result, filename, onAgain, user }) {
  const ratio = result.score / result.total;
  const message =
    ratio === 1 ? "Full marks."
      : ratio >= 0.8 ? "Strong result."
        : ratio >= 0.5 ? "Good start — the explanations show what to revisit."
          : "Worth another read through, then try a fresh set.";
  return (
    <div className="card pad quiz-score">
      <div className="quiz-score-n">{result.score}<span>/{result.total}</span></div>
      <p className="quiz-score-t">{message}</p>
      <p className="quiz-score-s">
        {filename}{user ? " · saved to your progress" : ""}
      </p>
      <button type="button" className="btn btn-primary" onClick={onAgain}>
        <RotateCcw aria-hidden="true" />Try a fresh set
      </button>
    </div>
  );
}

function Shell({ children }) {
  return (
    <div className="body">
      <div className="pane study">
        <h1 className="page">Quiz</h1>
        <p className="page-sub">Test yourself on one document at a time.</p>
        {children}
      </div>
    </div>
  );
}
