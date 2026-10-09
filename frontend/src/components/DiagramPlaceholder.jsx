/**
 * Stands in for a diagram while the answer is still streaming.
 *
 * A diagram arrives a few characters at a time and no intermediate state
 * is renderable - half a JSON object, or Mermaid missing its closing
 * edges - so something has to occupy the space until the stream ends.
 * It used to be the raw source, which meant the student watched Mermaid
 * arrows or JSON scroll past: the machinery, not the answer.
 *
 * This shows that a diagram is coming and nothing else. The fixed height
 * also stops the bubble jolting when the real diagram replaces it.
 */
export default function DiagramPlaceholder() {
  return (
    <div className="flow-placeholder" role="status" aria-live="polite">
      <span className="flow-placeholder-bar" aria-hidden="true" />
      <span className="flow-placeholder-text">Drawing diagram</span>
    </div>
  );
}
