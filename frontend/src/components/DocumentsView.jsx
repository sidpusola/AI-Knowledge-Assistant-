import { useRef, useState } from "react";
import { FileText, Link2, Loader2, Search, Trash2, Upload } from "lucide-react";

import { deleteDocument, ingestUrl, uploadFile } from "../api.js";

const ACCEPT = ".txt,.md,.pdf,.docx,.csv";

export default function DocumentsView({ documents, onDocumentsChanged }) {
  const [busy, setBusy] = useState(null);     // "upload" | "url" | a doc_id
  const [status, setStatus] = useState(null); // { tone, text }
  const [url, setUrl] = useState("");
  const [dragging, setDragging] = useState(false);
  const [filter, setFilter] = useState("");
  const fileInput = useRef(null);

  const say = (tone, text) => setStatus({ tone, text });

  const addFile = async (file) => {
    if (!file) return;
    setBusy("upload");
    setStatus(null);
    try {
      const data = await uploadFile(file);
      say("ok", data.message);
      onDocumentsChanged?.();
    } catch (error) {
      say("bad", error.message || "Upload failed.");
    } finally {
      setBusy(null);
    }
  };

  const addUrl = async (event) => {
    event.preventDefault();
    const value = url.trim();
    if (!value) return;
    setBusy("url");
    setStatus(null);
    try {
      const data = await ingestUrl(value);
      say("ok", data.message || "Page added.");
      setUrl("");
      onDocumentsChanged?.();
    } catch (error) {
      say("bad", error.message || "Could not read that page.");
    } finally {
      setBusy(null);
    }
  };

  const remove = async (doc) => {
    // Removing material students rely on shouldn't be a single misclick.
    const ok = window.confirm(
      `Remove "${doc.filename}" from the knowledge base?\n\n` +
        "Students will no longer get answers from it. The uploaded file itself is kept."
    );
    if (!ok) return;

    setBusy(doc.doc_id);
    setStatus(null);
    try {
      const data = await deleteDocument(doc.doc_id);
      say("ok", data.message);
      onDocumentsChanged?.();
    } catch (error) {
      say("bad", error.message || `Could not remove "${doc.filename}".`);
    } finally {
      setBusy(null);
    }
  };

  const needle = (filter || "").trim().toLowerCase();
  const shown = needle
    ? documents.filter((doc) =>
        `${doc.filename} ${doc.category} ${doc.file_type}`.toLowerCase().includes(needle)
      )
    : documents;

  const chunks = documents.reduce((total, doc) => total + (doc.chunks || 0), 0);

  return (
    <div className="body">
      <div className="pane">
        <h1 className="page">Documents</h1>
        <p className="page-sub">
          Everything here is searchable from Ask. {documents.length}{" "}
          {documents.length === 1 ? "document is" : "documents are"} indexed as{" "}
          {chunks} searchable {chunks === 1 ? "passage" : "passages"}.
        </p>

        <div
          className={`drop ${dragging ? "is-over" : ""}`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragging(false);
            addFile(event.dataTransfer.files?.[0]);
          }}
        >
          <div className="drop-ic"><Upload aria-hidden="true" /></div>
          <div className="drop-t">Drop a handout here, or choose a file</div>
          <div className="drop-s">PDF, Word, Markdown, plain text or CSV.</div>

          <input
            ref={fileInput}
            type="file"
            accept={ACCEPT}
            className="visually-hidden"
            onChange={(event) => {
              addFile(event.target.files?.[0]);
              event.target.value = "";
            }}
          />
          <button
            type="button"
            className="btn btn-primary btn-sm"
            style={{ marginTop: 14 }}
            onClick={() => fileInput.current?.click()}
            disabled={busy === "upload"}
          >
            {busy === "upload" ? (
              <><Loader2 className="spin" aria-hidden="true" />Adding…</>
            ) : (
              "Choose a file"
            )}
          </button>

          <form className="url-row" onSubmit={addUrl}>
            <label className="url-in">
              <Link2 aria-hidden="true" />
              <input
                type="url"
                value={url}
                onChange={(event) => setUrl(event.target.value)}
                placeholder="Paste a web page address instead"
                aria-label="Web page address"
              />
            </label>
            <button
              type="submit"
              className="btn btn-ghost"
              style={{ height: 38 }}
              disabled={!url.trim() || busy === "url"}
            >
              {busy === "url" ? "Reading…" : "Add page"}
            </button>
          </form>
        </div>

        {status && (
          <p className={`doc-status ${status.tone}`} role="status">{status.text}</p>
        )}

        {documents.length > 1 && (
          <label className="doc-filter">
            <Search aria-hidden="true" />
            <input
              type="search"
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
              placeholder="Filter by name or subject"
              aria-label="Filter documents"
            />
          </label>
        )}

        <div className="card pad">
          {documents.length === 0 ? (
            <p className="doc-empty">
              Nothing indexed yet. Add your first handout above and it becomes
              answerable straight away.
            </p>
          ) : shown.length === 0 ? (
            <p className="doc-empty">No document matches “{filter}”.</p>
          ) : (
            <table className="docs">
              <thead>
                <tr>
                  <th>Document</th><th>Type</th><th>Subject</th>
                  <th>Passages</th><th>Added</th><th />
                </tr>
              </thead>
              <tbody>
                {shown.map((doc) => (
                  <tr key={doc.doc_id}>
                    <td>
                      <div className="doc-name">
                        {doc.file_type === "web"
                          ? <Link2 aria-hidden="true" />
                          : <FileText aria-hidden="true" />}
                        {doc.filename}
                      </div>
                    </td>
                    <td>{doc.file_type}</td>
                    <td><span className="tag">{doc.category}</span></td>
                    <td>{doc.chunks}</td>
                    <td>{doc.ingested_at || "—"}</td>
                    <td>
                      <div className="row-acts">
                        <button
                          type="button"
                          className="btn btn-ghost btn-sm"
                          style={{ padding: "0 9px" }}
                          onClick={() => remove(doc)}
                          disabled={busy === doc.doc_id}
                          title={`Remove ${doc.filename}`}
                          aria-label={`Remove ${doc.filename}`}
                        >
                          <Trash2 aria-hidden="true" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </div>
    </div>
  );
}
