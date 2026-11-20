import { useEffect, useState } from "react";
import { Check, RefreshCw, X } from "lucide-react";

import {
  decideReview, fetchEvaluation, fetchFeedbackSummary, fetchReviews, fetchStats,
  fetchWorkflowGraph,
} from "../api.js";
import { pct } from "../lib/study.js";
import MermaidDiagram from "./MermaidDiagram.jsx";
import { ErrorNote, Working } from "./StudyCommon.jsx";

const seconds = (value) => (value == null ? "—" : `${Number(value).toFixed(1)}s`);

export default function MonitoringView({ user }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [version, setVersion] = useState(0);

  useEffect(() => {
    if (!user?.is_staff) return undefined;
    let cancelled = false;
    (async () => {
      try {
        // Each panel stands alone: one failing call should not blank the
        // whole dashboard.
        const settle = (promise) => promise.catch((err) => ({ error: err.message }));
        const [stats, feedback, evaluation, reviews, graph] = await Promise.all([
          settle(fetchStats()), settle(fetchFeedbackSummary()), settle(fetchEvaluation()),
          settle(fetchReviews()), settle(fetchWorkflowGraph()),
        ]);
        if (!cancelled) setData({ stats, feedback, evaluation, reviews, graph });
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [user, version]);

  if (!user?.is_staff) {
    return (
      <Shell>
        <div className="card pad study-intro"><p>Monitoring is available to course staff only.</p></div>
      </Shell>
    );
  }

  return (
    <Shell onRefresh={() => setVersion((v) => v + 1)}>
      <ErrorNote message={error} />
      {!data && !error && <Working label="Loading monitoring data…" />}
      {data && (
        <>
          <LivePanel stats={data.stats} feedback={data.feedback} reviews={data.reviews} />
          <ReviewQueue reviews={data.reviews} onDecided={() => setVersion((v) => v + 1)} />
          <EvaluationPanel evaluation={data.evaluation} />
          {data.graph?.mermaid && (
            <section className="card pad">
              <h2 className="guide-h">Controlled workflow</h2>
              <p className="guide-sources">The LangGraph state machine behind /workflow, drawn from the compiled graph.</p>
              <MermaidDiagram rawCode={data.graph.mermaid} />
            </section>
          )}
        </>
      )}
    </Shell>
  );
}

function LivePanel({ stats, feedback, reviews }) {
  const latency = stats?.latency_seconds || {};
  const pending = reviews?.reviews?.length ?? 0;
  return (
    <div className="tiles">
      <Tile label="Answers served" value={stats?.total_requests ?? "—"}
        sub={stats?.error ? stats.error : `${stats?.errors_in_sample ?? 0} errors in the last ${stats?.recent_sample_size ?? 0}`} />
      <Tile label="Answer time" value={seconds(latency.avg_total)}
        sub={`p95 ${seconds(latency.p95_total)} · first token ${seconds(latency.avg_time_to_first_token)}`} />
      <Tile label="Student satisfaction" value={pct(feedback?.satisfaction)}
        sub={feedback?.total ? `${feedback.up} up · ${feedback.down} down` : "No ratings yet"} />
      <Tile label="Awaiting review" value={pending}
        sub={pending ? "Answers held by the workflow" : "Nothing held"} />
    </div>
  );
}

function ReviewQueue({ reviews, onDecided }) {
  const items = reviews?.reviews || [];
  return (
    <section className="card pad">
      <h2 className="guide-h">Human review queue</h2>
      <p className="guide-sources">
        Answers the workflow could not verify against the materials after a retry. Approve
        (editing if needed) to release one to the student, or reject it.
      </p>
      {reviews?.error && <ErrorNote message={reviews.error} />}
      {!items.length && !reviews?.error && <p className="doc-empty">No answers are waiting.</p>}
      {items.map((item) => <ReviewItem key={item.thread_id} item={item} onDecided={onDecided} />)}
    </section>
  );
}

function ReviewItem({ item, onDecided }) {
  const [draft, setDraft] = useState(item.draft);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const decide = async (approved) => {
    setBusy(true);
    setError(null);
    try {
      await decideReview(item.thread_id, { approved, answer: approved ? draft : "" });
      onDecided();
    } catch (err) {
      setError(err.message);
      setBusy(false);
    }
  };

  return (
    <article className="review">
      <div className="review-q">{item.question}</div>
      {item.issues.length > 0 && (
        <ul className="review-issues">
          {item.issues.map((issue) => <li key={issue}>Unsupported: {issue}</li>)}
        </ul>
      )}
      <label className="iv-answer">
        <span className="visually-hidden">Draft answer</span>
        <textarea value={draft} onChange={(e) => setDraft(e.target.value)} rows={4} disabled={busy} />
      </label>
      <div className="review-foot">
        <span className="guide-sources">{item.sources.join("; ") || "No sources"}</span>
        <div className="guide-cta">
          <button type="button" className="btn btn-ghost btn-sm" onClick={() => decide(false)} disabled={busy}>
            <X aria-hidden="true" />Reject
          </button>
          <button type="button" className="btn btn-primary btn-sm" onClick={() => decide(true)} disabled={busy || !draft.trim()}>
            <Check aria-hidden="true" />Approve
          </button>
        </div>
      </div>
      <ErrorNote message={error} />
    </article>
  );
}

const METRICS = [
  ["pass_rate", "Cases passed"],
  ["route_accuracy", "Classification accuracy"],
  ["retrieval_relevance", "Retrieval relevance"],
  ["answer_correctness", "Answer correctness"],
  ["faithfulness", "Faithfulness"],
  ["hallucination_control", "Hallucination control"],
  ["injection_resistance", "Injection resistance"],
  ["held_for_review", "Held for review"],
];

function EvaluationPanel({ evaluation }) {
  if (evaluation?.error) return <ErrorNote message={evaluation.error} />;
  if (!evaluation?.available) {
    return (
      <section className="card pad">
        <h2 className="guide-h">Evaluation</h2>
        <p className="doc-empty">No evaluation has been run yet. Run <code>python -m tests.rag_eval</code>.</p>
      </section>
    );
  }
  const s = evaluation.summary;
  return (
    <section className="card pad">
      <h2 className="guide-h">Evaluation</h2>
      <p className="guide-sources">
        Last run {new Date(evaluation.run_at).toLocaleString()} · {s.passed}/{s.cases} cases ·
        {" "}{Object.values(evaluation.prompts).join(", ")}
      </p>
      <dl className="metric-grid">
        {METRICS.map(([key, label]) => (
          <div key={key}><dt>{label}</dt><dd>{pct(s[key])}</dd></div>
        ))}
      </dl>
      <div className="table-scroll">
        <table className="docs stat-table">
          <thead><tr><th>Category</th><th>Cases</th><th>Pass rate</th></tr></thead>
          <tbody>
            {Object.entries(s.by_category).map(([name, v]) => (
              <tr key={name}><td>{name.replace("_", " ")}</td><td className="num">{v.cases}</td><td className="num">{pct(v.pass_rate)}</td></tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function Tile({ label, value, sub }) {
  return (
    <div className="card tile">
      <div className="tile-label">{label}</div>
      <div className="tile-value">{value}</div>
      <div className="tile-sub">{sub}</div>
    </div>
  );
}

function Shell({ children, onRefresh }) {
  return (
    <div className="body">
      <div className="pane study">
        <div className="study-bar" style={{ justifyContent: "space-between" }}>
          <div>
            <h1 className="page">Monitoring</h1>
            <p className="page-sub" style={{ marginBottom: 0 }}>Live usage, answer quality and the review queue.</p>
          </div>
          {onRefresh && (
            <button type="button" className="btn btn-ghost btn-sm" onClick={onRefresh}>
              <RefreshCw aria-hidden="true" />Refresh
            </button>
          )}
        </div>
        {children}
      </div>
    </div>
  );
}
