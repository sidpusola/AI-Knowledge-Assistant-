import { useEffect, useState } from "react";
import { listDocuments, uploadFile } from "../api.js";

export default function KnowledgeBase({ refreshKey, onDocumentsChanged }) {
  // null = not loaded yet, so "loading" is derived from state rather than
  // tracked with a separate flag set synchronously inside an effect.
  const [documents, setDocuments] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  const [pendingFile, setPendingFile] = useState(null);

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
            {documents.map((name) => (
              <li key={name} className="kb-list-item">
                <span className="kb-file-icon">📄</span>
                <span className="kb-file-name">{name}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
