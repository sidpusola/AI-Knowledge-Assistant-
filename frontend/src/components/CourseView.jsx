import { useEffect, useState } from "react";
import { Briefcase, Compass, FileText, Layers, Link2, ListChecks } from "lucide-react";

import { fetchProgress } from "../api.js";
import { outOfFive, pct } from "../lib/study.js";
import { NoDocuments } from "./StudyCommon.jsx";

export default function CourseView({ documents, user, onNavigate }) {
  const [perDoc, setPerDoc] = useState({});

  useEffect(() => {
    if (!user) return undefined;
    let cancelled = false;
    (async () => {
      try {
        const data = await fetchProgress();
        if (!cancelled) {
          setPerDoc(Object.fromEntries(data.documents.map((d) => [d.doc_id, d])));
        }
      } catch {
        // Progress decorates the course list; the list works without it.
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [user]);

  const groups = {};
  for (const doc of documents) {
    const key = doc.category || "uncategorised";
    (groups[key] ||= []).push(doc);
  }
  const subjects = Object.keys(groups).sort();

  return (
    <div className="body">
      <div className="pane study">
        <h1 className="page">My course</h1>
        <p className="page-sub">
          {documents.length
            ? `${documents.length} ${documents.length === 1 ? "document" : "documents"} across ${subjects.length} ${subjects.length === 1 ? "subject" : "subjects"}. Pick one to study or practise.`
            : "Your course material, grouped by subject."}
        </p>

        {!documents.length && <NoDocuments onNavigate={onNavigate} />}

        {subjects.map((subject) => (
          <section key={subject} className="course-group">
            <h2 className="course-subject">{subject}</h2>
            <div className="course-grid">
              {groups[subject].map((doc) => (
                <CourseCard
                  key={doc.doc_id}
                  doc={doc}
                  stats={perDoc[doc.doc_id]}
                  signedIn={Boolean(user)}
                  onNavigate={onNavigate}
                />
              ))}
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}

function CourseCard({ doc, stats, signedIn, onNavigate }) {
  const go = (tab) => onNavigate(tab, doc.doc_id);
  return (
    <article className="card course-card">
      <div className="course-name">
        {doc.file_type === "web" ? <Link2 aria-hidden="true" /> : <FileText aria-hidden="true" />}
        <span>{doc.filename}</span>
      </div>
      <div className="course-meta">
        {doc.chunks} passages{doc.ingested_at ? ` · added ${doc.ingested_at}` : ""}
      </div>

      {signedIn && (
        <dl className="course-stats">
          <div><dt>Best quiz</dt><dd>{pct(stats?.quiz_best)}</dd></div>
          <div><dt>Cards known</dt><dd>{pct(stats?.cards_known)}</dd></div>
          <div><dt>Interview</dt><dd>{outOfFive(stats?.interview_avg)}</dd></div>
        </dl>
      )}

      <div className="course-acts">
        <button type="button" className="btn btn-ghost btn-sm" onClick={() => go("guide")}>
          <Compass aria-hidden="true" />Guide
        </button>
        <button type="button" className="btn btn-ghost btn-sm" onClick={() => go("quiz")}>
          <ListChecks aria-hidden="true" />Quiz
        </button>
        <button type="button" className="btn btn-ghost btn-sm" onClick={() => go("cards")}>
          <Layers aria-hidden="true" />Cards
        </button>
        <button type="button" className="btn btn-ghost btn-sm" onClick={() => go("placement")}>
          <Briefcase aria-hidden="true" />Interview
        </button>
      </div>
    </article>
  );
}
