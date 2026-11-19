import { useEffect, useState } from "react";
import { BarChart3 } from "lucide-react";

import { fetchProgress } from "../api.js";
import { outOfFive, pct } from "../lib/study.js";
import { ErrorNote, Working } from "./StudyCommon.jsx";

const KIND_LABEL = { quiz: "Quiz", flashcards: "Flashcards", interview: "Interview" };

export default function ProgressView({ user }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!user) return undefined;
    let cancelled = false;
    (async () => {
      try {
        const progress = await fetchProgress();
        if (!cancelled) setData(progress);
      } catch (err) {
        if (!cancelled) setError(err.message);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [user]);

  return (
    <div className="body">
      <div className="pane study">
        <h1 className="page">Progress</h1>
        <p className="page-sub">What you've asked and practised — counted from your own activity, nothing estimated.</p>

        {!user && (
          <div className="card pad study-intro">
            <p>
              Sign in from the menu at the bottom of the sidebar to see your
              progress. Questions you ask and practice you do while signed in
              are saved here.
            </p>
          </div>
        )}

        <ErrorNote message={error} />
        {user && !data && !error && <Working label="Loading your progress…" />}
        {data && <Dashboard data={data} />}
      </div>
    </div>
  );
}

/* The server stores UTC without a zone suffix (SQLite keeps none). */
const asUtc = (stamp) => new Date(/(Z|[+-]\d\d:\d\d)$/.test(stamp) ? stamp : `${stamp}Z`);

function Dashboard({ data }) {
  return (
    <>
      <div className="tiles">
        <Tile label="Questions asked" value={data.questions_asked.toLocaleString()}
          sub={`in ${data.conversations} ${data.conversations === 1 ? "conversation" : "conversations"}`} />
        <Tile label="Quiz average" value={pct(data.quiz.ratio)}
          sub={`${data.quiz.count} ${data.quiz.count === 1 ? "quiz" : "quizzes"} taken`} />
        <Tile label="Flashcards known" value={pct(data.flashcards.ratio)}
          sub={`${data.flashcards.out_of} cards reviewed, self-rated`} />
        <Tile label="Interview average"
          value={outOfFive(data.interview.count ? data.interview.score / data.interview.count : null)}
          sub={`${data.interview.count} ${data.interview.count === 1 ? "answer" : "answers"} scored`} />
      </div>

      <ActivityChart days={data.activity} />

      <section className="card pad">
        <h2 className="guide-h">By document</h2>
        {data.documents.length === 0 ? (
          <p className="doc-empty">No practice yet. Try a quiz or a deck of flashcards from My course.</p>
        ) : (
          <div className="table-scroll">
            <table className="docs stat-table progress-table">
              <thead>
                <tr>
                  <th>Document</th><th>Quizzes</th><th>Best</th><th>Latest</th>
                  <th>Cards known</th><th>Interview</th>
                </tr>
              </thead>
              <tbody>
                {data.documents.map((doc) => (
                  <tr key={doc.doc_id}>
                    <td><div className="doc-name">{doc.filename}</div></td>
                    <td className="num">{doc.quizzes || "—"}</td>
                    <td className="num">{pct(doc.quiz_best)}</td>
                    <td className="num">{pct(doc.quiz_last)}</td>
                    <td className="num">{pct(doc.cards_known)}</td>
                    <td className="num">{outOfFive(doc.interview_avg)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {data.recent.length > 0 && (
        <section className="card pad">
          <h2 className="guide-h">Recent practice</h2>
          <ul className="recent">
            {data.recent.map((row, i) => (
              <li key={`${row.created_at}-${i}`}>
                <span className="tag">{KIND_LABEL[row.kind] || row.kind}</span>
                <span className="recent-doc">{row.filename}</span>
                <span className="recent-score">{row.score}/{row.out_of}</span>
                <time className="recent-when" dateTime={row.created_at}>
                  {asUtc(row.created_at).toLocaleDateString(undefined, { day: "numeric", month: "short" })}
                </time>
              </li>
            ))}
          </ul>
        </section>
      )}
    </>
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

/* Round a maximum up to 1, 2 or 5 times a power of ten, so the axis
 * reads 0 / 5 / 10 rather than 0 / 3.5 / 7. */
function niceCeiling(max) {
  if (max <= 1) return 1;
  const power = 10 ** Math.floor(Math.log10(max));
  const step = [1, 2, 5, 10].find((m) => m * power >= max);
  return step * power;
}

const SERIES = [
  { key: "questions", label: "Questions asked" },
  { key: "practice", label: "Practice sessions" },
];

const PLOT_HEIGHT = 150;

function ActivityChart({ days }) {
  const [hover, setHover] = useState(null);
  const [asTable, setAsTable] = useState(false);

  const totals = days.map((d) => d.questions + d.practice);
  const top = niceCeiling(Math.max(...totals, 0));
  const empty = totals.every((t) => t === 0);
  const fmt = (iso, opts) => new Date(`${iso}T00:00:00`).toLocaleDateString(undefined, opts);

  return (
    <section className="card pad">
      <div className="chart-head">
        <div>
          <h2 className="guide-h">Last 14 days</h2>
          <div className="chart-legend">
            {SERIES.map((s, i) => (
              <span key={s.key}><i className={`key s${i + 1}`} aria-hidden="true" />{s.label}</span>
            ))}
          </div>
        </div>
        <button type="button" className="btn btn-quiet btn-sm" onClick={() => setAsTable((v) => !v)}>
          {asTable ? "Show chart" : "Show table"}
        </button>
      </div>

      {empty && !asTable ? (
        <p className="doc-empty chart-empty">
          <BarChart3 aria-hidden="true" />No activity in the last 14 days yet.
        </p>
      ) : asTable ? (
        <div className="table-scroll">
          <table className="docs stat-table">
            <thead><tr><th>Day</th><th>Questions asked</th><th>Practice sessions</th></tr></thead>
            <tbody>
              {days.map((d) => (
                <tr key={d.date}>
                  <td>{fmt(d.date, { weekday: "short", day: "numeric", month: "short" })}</td>
                  <td className="num">{d.questions}</td>
                  <td className="num">{d.practice}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <div className="act">
          <div className="act-axis" aria-hidden="true">
            {[top, top / 2, 0].map((tick) => (
              <span key={tick} style={{ bottom: `${(tick / top) * PLOT_HEIGHT}px` }}>
                {Number.isInteger(tick) ? tick : tick.toFixed(1)}
              </span>
            ))}
          </div>
          <div className="act-plot" style={{ height: PLOT_HEIGHT }}>
            {[top, top / 2, 0].map((tick) => (
              <div key={tick} className="act-grid" style={{ bottom: `${(tick / top) * PLOT_HEIGHT}px` }} />
            ))}
            {days.map((d, i) => {
              const qh = (d.questions / top) * PLOT_HEIGHT;
              const ph = (d.practice / top) * PLOT_HEIGHT;
              const label = `${fmt(d.date, { weekday: "short", day: "numeric", month: "short" })}: `
                + `${d.questions} questions asked, ${d.practice} practice sessions`;
              return (
                <div
                  key={d.date}
                  className={`act-col ${hover === i ? "is-hover" : ""}`}
                  tabIndex={0}
                  aria-label={label}
                  onPointerEnter={() => setHover(i)}
                  onPointerLeave={() => setHover(null)}
                  onFocus={() => setHover(i)}
                  onBlur={() => setHover(null)}
                >
                  <div className="act-bar">
                    {d.practice > 0 && (
                      <span className="seg s2 cap" style={{ height: ph }} />
                    )}
                    {d.questions > 0 && (
                      <span
                        className={`seg s1 ${d.practice > 0 ? "" : "cap"}`}
                        style={{ height: qh, marginTop: d.practice > 0 ? 2 : 0 }}
                      />
                    )}
                  </div>
                  {hover === i && (
                    <div className={`act-tip ${i > days.length / 2 ? "left" : ""}`} role="tooltip">
                      <div className="act-tip-d">{fmt(d.date, { weekday: "short", day: "numeric", month: "short" })}</div>
                      {SERIES.map((s, n) => (
                        <div key={s.key} className="act-tip-r">
                          <i className={`line s${n + 1}`} aria-hidden="true" />
                          <strong>{d[s.key]}</strong><span>{s.label.toLowerCase()}</span>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
          <div className="act-x" aria-hidden="true">
            {days.map((d, i) => (
              <span key={d.date}>
                {i === 0 || d.date.endsWith("-01") ? fmt(d.date, { day: "numeric", month: "short" }) : Number(d.date.slice(8))}
              </span>
            ))}
          </div>
        </div>
      )}
    </section>
  );
}
