import { useEffect, useState } from "react";
import { deleteDocument, listDocuments, uploadFile } from "../api.js";

export default function KnowledgeBase({ refreshKey, onDocumentsChanged }) {
  // null = not loaded yet, so "loading" is derived from state rather than
  // tracked with a separate flag set synchronously inside an effect.
  const [documents, setDocuments] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [pendingFile, setPendingFile] = useState(null);
  const [deletingId, setDeletingId] = useState(null);

  const refresh = async () => {
    try {
      const data = await listDocuments();
      setDocuments(data.documents || []);
    } catch {
      setStatusMessage("Could not load the knowledge base. Is the server running?");
      setDocuments((prev) => prev ?? []);
    }
  };

  useEffect(() => {
    (async () => {
      await refresh();
    })();
  }, [refreshKey]);

  const handleDelete = async (document) => {
    // Removing material students rely on shouldn't be a single misclick.
    const confirmed = window.confirm(
      `Remove "${document.filename}" from the knowledge base?\n\n` +
        `Students will no longer get answers from it. The uploaded file itself is kept.`
    );
    if (!confirmed) return;

    setDeletingId(document.doc_id);
    setStatusMessage("");

    try {
      const data = await deleteDocument(document.doc_id);
      setStatusMessage(data.message);
      await refresh();
      onDocumentsChanged?.();
    } catch (error) {
      setStatusMessage(error.message || `Could not remove "${document.filename}".`);
    } finally {
      setDeletingId(null);
    }
  };

  const handleUpload = async () => {
    if (!pendingFile) return;
    setIsUploading(true);
    setStatusMessage("");

    try {
      const data = await uploadFile(pendingFile);
      setStatusMessage(data.message);
      setPendingFile(null);
      await refresh();
      onDocumentsChanged?.();
    } catch (error) {
      setStatusMessage(error.message || "Upload failed.");
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="kb-view">
      <div className="kb-intro">
        <h2>Knowledge base</h2>
        <p>
          Materials added here are available to every student in every conversation.
          Use this to curate the course content the assistant answers from.
        </p>
      </div>

      <div className="kb-upload">
        <input
          type="file"
          id="kb-file-input"
          className="kb-file-input"
          accept=".txt,.md,.pdf,.docx,.csv"
          onChange={(event) => setPendingFile(event.target.files?.[0] || null)}
        />
        <label htmlFor="kb-file-input" className="kb-file-label">
          {pendingFile ? pendingFile.name : "Choose a file to add…"}
        </label>
        <button
          type="button"
          onClick={handleUpload}
          disabled={!pendingFile || isUploading}
        >
          {isUploading ? "Adding…" : "Add to knowledge base"}
        </button>
      </div>

      {statusMessage && <p className="kb-status">{statusMessage}</p>}

      <div className="kb-list">
        {documents === null ? (
          <p className="kb-empty">Loading…</p>
        ) : documents.length === 0 ? (
          <p className="kb-empty">No materials yet — add your first document above.</p>
        ) : (
          <ul>
            {documents.map((document) => (
              <li key={document.doc_id} className="kb-list-item">
                <span className="kb-file-icon">
                  {document.file_type === "web" ? "🌐" : "📄"}
                </span>

                <span className="kb-file-main">
                  <span className="kb-file-name">{document.filename}</span>
                  <span className="kb-file-meta">
                    {document.category} · {document.chunks} chunk
                    {document.chunks === 1 ? "" : "s"}
                    {document.ingested_at ? ` · added ${document.ingested_at}` : ""}
                  </span>
                </span>

                <button
                  type="button"
                  className="kb-remove"
                  title={`Remove ${document.filename} from the knowledge base`}
                  onClick={() => handleDelete(document)}
                  disabled={deletingId === document.doc_id}
                >
                  {deletingId === document.doc_id ? "…" : "Remove"}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
