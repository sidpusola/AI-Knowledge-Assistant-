import { useState } from "react";
import { BookOpen, RotateCcw } from "lucide-react";

import { makeFlashcards, recordFlashcards } from "../api.js";
import { initialDoc } from "../lib/study.js";
import { DocPicker, ErrorNote, NoDocuments, SignInNote, Working } from "./StudyCommon.jsx";

export default function FlashcardsView({ documents, focusDoc, user, onNavigate }) {
  const [docId, setDocId] = useState(() => initialDoc(documents, focusDoc));
  const [deck, setDeck] = useState(null);        // { filename, cards }
  const [queue, setQueue] = useState([]);        // indexes into deck.cards still to go
  const [flipped, setFlipped] = useState(false);
  const [known, setKnown] = useState([]);
  const [missed, setMissed] = useState([]);
  const [firstPass, setFirstPass] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  if (!documents.length) return <Shell><NoDocuments onNavigate={onNavigate} /></Shell>;

  const chosenDoc = docId || initialDoc(documents, focusDoc);

  const reset = (cards) => {
    setQueue(cards.map((_, i) => i));
    setKnown([]);
    setMissed([]);
    setFlipped(false);
  };

  const build = async () => {
    setBusy(true);
    setError(null);
    setDeck(null);
    try {
      const data = await makeFlashcards(chosenDoc, 8);
      setDeck(data);
      setFirstPass(true);
      reset(data.cards);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const mark = (gotIt) => {
    const [current, ...rest] = queue;
    const nextKnown = gotIt ? [...known, current] : known;
    const nextMissed = gotIt ? missed : [...missed, current];
    setKnown(nextKnown);
    setMissed(nextMissed);
    setQueue(rest);
    setFlipped(false);

    // Only the first time through a deck counts. Going round again on the
    // missed cards is practice, and would otherwise inflate the score.
    if (!rest.length && firstPass && user) {
      recordFlashcards(chosenDoc, nextKnown.length, deck.cards.length).catch(() => {});
    }
  };

  const retryMissed = () => {
    setFirstPass(false);
    setQueue(missed);
    setKnown([]);
    setMissed([]);
    setFlipped(false);
  };

  const card = deck && queue.length ? deck.cards[queue[0]] : null;
  const done = deck && !queue.length;
  const reviewed = known.length + missed.length;

  return (
    <Shell>
      <div className="study-bar">
        <DocPicker
          documents={documents}
          value={chosenDoc}
          onChange={(value) => { setDocId(value); setDeck(null); }}
          disabled={busy}
        />
        <button type="button" className="btn btn-primary" onClick={build} disabled={!chosenDoc || busy}>
          {deck ? "New deck" : "Make flashcards"}
        </button>
      </div>

      <SignInNote user={user} what="your results" />
      <ErrorNote message={error} />

      {busy && (
        <Working
          label="Making flashcards from this document…"
          hint="Eight cards, each from a different passage. Usually about ten seconds."
        />
      )}

      {!deck && !busy && (
        <div className="card pad study-intro">
          <p>
            Key terms and ideas from your document, one card at a time. Try to
            answer before you flip, then be honest about whether you knew it —
            the ones you missed come back round.
          </p>
        </div>
      )}

      {card && (
        <>
          <div className="fc-meta">
            Card {reviewed + 1} of {reviewed + queue.length}
            {!firstPass && <span className="tag">missed cards</span>}
          </div>
          <button
            type="button"
            className={`fc-card ${flipped ? "is-flipped" : ""}`}
            onClick={() => setFlipped((f) => !f)}
            aria-label={flipped ? "Show the question side" : "Show the answer"}
          >
            <span className="fc-face">
              <span className="fc-side">{flipped ? "Answer" : "Term or question"}</span>
              <span className={flipped ? "fc-back" : "fc-front"}>{flipped ? card.back : card.front}</span>
              {flipped && (
                <span className="fc-src"><BookOpen aria-hidden="true" />{card.source}</span>
              )}
              {!flipped && <span className="fc-hint">Click to flip</span>}
            </span>
          </button>
          <div className="fc-actions">
            {flipped ? (
              <>
                <button type="button" className="btn btn-ghost" onClick={() => mark(false)}>Still learning</button>
                <button type="button" className="btn btn-primary" onClick={() => mark(true)}>I knew it</button>
              </>
            ) : (
              <button type="button" className="btn btn-ghost" onClick={() => setFlipped(true)}>Show answer</button>
            )}
          </div>
        </>
      )}

      {done && (
        <div className="card pad quiz-score">
          <div className="quiz-score-n">{known.length}<span>/{known.length + missed.length}</span></div>
          <p className="quiz-score-t">
            {missed.length ? `${missed.length} still to learn.` : "You knew every card."}
          </p>
          <p className="quiz-score-s">
            {deck.filename}{firstPass && user ? " · saved to your progress" : ""}
          </p>
          <div className="quiz-actions center">
            {missed.length > 0 && (
              <button type="button" className="btn btn-primary" onClick={retryMissed}>
                <RotateCcw aria-hidden="true" />Go over the missed ones
              </button>
            )}
            <button type="button" className="btn btn-ghost" onClick={build}>New deck</button>
          </div>
        </div>
      )}
    </Shell>
  );
}

function Shell({ children }) {
  return (
    <div className="body">
      <div className="pane study">
        <h1 className="page">Flashcards</h1>
        <p className="page-sub">Revise key terms and ideas, and repeat the ones that didn't stick.</p>
        {children}
      </div>
    </div>
  );
}
