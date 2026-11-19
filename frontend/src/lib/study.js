/** The document a screen starts on: the one handed over from another
 *  screen, else the first in the list. */
export function initialDoc(documents, preferred) {
  if (preferred && documents.some((doc) => doc.doc_id === preferred)) return preferred;
  return documents[0]?.doc_id || "";
}

export const pct = (ratio) => (ratio == null ? "—" : `${Math.round(ratio * 100)}%`);

export const outOfFive = (score) => (score == null ? "—" : `${Number(score).toFixed(1)}/5`);
