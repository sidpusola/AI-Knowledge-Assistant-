import { useEffect, useState } from "react";
import { BookOpen, Layers, ListChecks, RefreshCw } from "lucide-react";

import { fetchGuide } from "../api.js";
import { initialDoc } from "../lib/study.js";
import { DocPicker, ErrorNote, NoDocuments, Working } from "./StudyCommon.jsx";

export default function GuideView({ documents, focusDoc, onNavigate }) {
  const [request, setRequest] = useState({ docId: "", refresh: false });
  const [guide, setGuide] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const docId = request.docId || initialDoc(documents, focusDoc);

  useEffect(() => {
    if (!docId) return undefined;
    let cancelled = false;
    (async () => {
      setBusy(true);
      setError(null);
      setGuide(null);
      try {
        const data = await fetchGuide(docId, { refresh: request.refresh });
        if (!cancelled) setGuide(data);
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setBusy(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [docId, request]);

  if (!documents.length) return <Shell><NoDocuments onNavigate={onNavigate} /></Shell>;

  return (
    <Shell>
      <div className="study-bar">
        <DocPicker
          documents={documents}
          value={docId}
          onChange={(value) => setRequest({ docId: value, refresh: false })}
          disabled={busy}
        />
        {guide && (
          <button
            type="button"
            className="btn btn-ghost"
            onClick={() => setRequest({ docId, refresh: true })}
            disabled={busy}
            title="Write the guide again from the document"
          >
            <RefreshCw aria-hidden="true" />Regenerate
          </button>
        )}
      </div>

      <ErrorNote message={error} />

      {busy && (
        <Working
          label="Preparing the study guide…"
          hint="The first time for a document takes 10–20 seconds; after that it opens instantly."
        />
      )}

      {guide && !busy && (
        <div className="guide">
          <section className="card pad">
            <h2 className="guide-h">Overview</h2>
            <p className="guide-overview">{guide.overview || "No overview could be written for this document."}</p>
            {guide.key_points.length > 0 && (
              <>
                <h3 className="guide-h3">Main points</h3>
                <ul className="guide-points">
                  {guide.key_points.map((point) => <li key={point}>{point}</li>)}
                </ul>
              </>
            )}
          </section>

          {guide.key_terms.length > 0 && (
            <section className="card pad">
              <h2 className="guide-h">Key terms</h2>
              <dl className="guide-terms">
                {guide.key_terms.map(({ term, meaning }) => (
                  <div key={term} className="guide-term">
                    <dt>{term}</dt>
                    <dd>{meaning}</dd>
                  </div>
                ))}
              </dl>
            </section>
          )}

          <section className="card pad guide-next">
            <div>
              <h2 className="guide-h">Check what stuck</h2>
              <p className="guide-sources">
                <BookOpen aria-hidden="true" />
                Written from {guide.sources.length} passages: {guide.sources.slice(0, 4).join("; ")}
                {guide.sources.length > 4 ? "…" : ""}
              </p>
            </div>
            <div className="guide-cta">
              <button type="button" className="btn btn-primary btn-sm" onClick={() => onNavigate("quiz", docId)}>
                <ListChecks aria-hidden="true" />Take a quiz
              </button>
              <button type="button" className="btn btn-ghost btn-sm" onClick={() => onNavigate("cards", docId)}>
                <Layers aria-hidden="true" />Flashcards
              </button>
            </div>
          </section>
        </div>
      )}
    </Shell>
  );
}

function Shell({ children }) {
  return (
    <div className="body">
      <div className="pane study">
        <h1 className="page">Study guide</h1>
        <p className="page-sub">The shape of a document before you dive in: what it covers and the terms to know.</p>
        {children}
      </div>
    </div>
  );
}
