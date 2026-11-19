import { FileText, Loader2 } from "lucide-react";

/** Every practice screen works on one document at a time. */
export function DocPicker({ documents, value, onChange, disabled }) {
  return (
    <label className="doc-pick">
      <FileText aria-hidden="true" />
      <select
        value={value || ""}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
        aria-label="Choose a document"
      >
        <option value="" disabled>Choose a document…</option>
        {documents.map((doc) => (
          <option key={doc.doc_id} value={doc.doc_id}>{doc.filename}</option>
        ))}
      </select>
    </label>
  );
}

export function NoDocuments({ onNavigate }) {
  return (
    <div className="card pad study-empty">
      <p>
        There is nothing to practise from yet. Add a handout in Documents and
        it becomes available here straight away.
      </p>
      <button type="button" className="btn btn-primary btn-sm" onClick={() => onNavigate("documents")}>
        Go to Documents
      </button>
    </div>
  );
}

/** Generation runs a local model several times over, so it takes real
 *  seconds. Saying roughly how long stops it reading as a hang. */
export function Working({ label, hint }) {
  return (
    <div className="card pad study-working" role="status" aria-live="polite">
      <Loader2 className="spin" aria-hidden="true" />
      <div>
        <div className="study-working-t">{label}</div>
        {hint && <div className="study-working-s">{hint}</div>}
      </div>
    </div>
  );
}

export function ErrorNote({ message }) {
  if (!message) return null;
  return <p className="doc-status bad" role="alert">{message}</p>;
}

export function SignInNote({ user, what }) {
  if (user) return null;
  return (
    <p className="study-note">
      You are not signed in, so {what} won't be saved to your progress.
    </p>
  );
}
