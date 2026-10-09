import { useCallback, useMemo, useRef, useState } from "react";
import { Check, Copy, Maximize2, Minimize2 } from "lucide-react";

import CardFlowchart from "./CardFlowchart.jsx";
import CycleDiagram from "./CycleDiagram.jsx";
import TimelineDiagram from "./TimelineDiagram.jsx";
import MermaidDiagram from "./MermaidDiagram.jsx";
import { parseFlowchart, resolveDiagramType, toMermaid } from "../lib/diagramSchema.js";

/**
 * Chooses how to draw a diagram, and owns the chrome around it.
 *
 * The decision is made from the block's CONTENT, not its fence name.
 * Fence names proved unreliable: "flowchart" is also Mermaid's own
 * keyword, so the model cheerfully wrote ```flowchart LR with Mermaid
 * syntax inside it, and once produced a hybrid of both. Content can't
 * lie the same way - JSON starts with a brace, Mermaid starts with a
 * diagram keyword - so the fence is only a hint and this sniffs.
 *
 *   starts with "{"  -> validated JSON. A plain sequence becomes cards;
 *                       anything that branches is converted to Mermaid,
 *                       because cards can only express one chain.
 *                       Invalid JSON falls back to the source with a
 *                       note, never a blank space.
 *   anything else    -> Mermaid, exactly as before this existed.
 */
/** Which renderer a block needs. Kept out of the component so the
 *  React Compiler can preserve the memo - it bails on a useMemo body
 *  with several early returns. */
function resolveDiagram(source) {
  if (!/^\s*\{/.test(source)) return { mode: "mermaid", code: source };

  const parsed = parseFlowchart(source);
  if (!parsed.ok) return { mode: "invalid", reason: parsed.reason };

  // cards, ring and timeline can each draw exactly one chain; anything
  // else is a graph, which is Mermaid's job.
  const mode = resolveDiagramType(parsed.diagram);
  if (mode === "mermaid") return { mode, code: toMermaid(parsed.diagram) };
  return { mode, diagram: parsed.diagram };
}

export default function FlowchartRenderer({ source }) {
  const frameRef = useRef(null);
  const [copied, setCopied] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const resolved = useMemo(() => resolveDiagram(source), [source]);

  const copy = useCallback(async () => {
    try {
      await navigator.clipboard.writeText(source);
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      // Clipboard access can be refused (insecure origin, denied
      // permission). Nothing useful to tell the user mid-chat, and the
      // source is selectable anyway.
    }
  }, [source]);

  const toggleFullscreen = useCallback(async () => {
    const node = frameRef.current;
    if (!node) return;
    try {
      if (document.fullscreenElement) {
        await document.exitFullscreen();
        setIsFullscreen(false);
      } else {
        await node.requestFullscreen();
        setIsFullscreen(true);
      }
    } catch {
      // Fullscreen is blocked in some embeddings; the diagram is still
      // perfectly readable inline.
    }
  }, []);

  return (
    <div className={`flow-frame flow-theme-${resolved.diagram?.theme || "minimal"}`} ref={frameRef}>
      <div className="flow-toolbar">
        <button
          type="button"
          className="flow-toolbar-button"
          onClick={copy}
          aria-label={copied ? "Diagram source copied" : "Copy diagram source"}
          title="Copy source"
        >
          {copied ? <Check size={14} /> : <Copy size={14} />}
        </button>
        <button
          type="button"
          className="flow-toolbar-button"
          onClick={toggleFullscreen}
          aria-label={isFullscreen ? "Exit fullscreen" : "Show diagram fullscreen"}
          title={isFullscreen ? "Exit fullscreen" : "Fullscreen"}
        >
          {isFullscreen ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
        </button>
      </div>

      {/* tabIndex makes the region reachable by keyboard, so a tall
          diagram that scrolls internally can be scrolled with the arrow
          keys rather than being a mouse-only trap. */}
      <div
        className="flow-scroll"
        tabIndex={0}
        role="group"
        aria-label={resolved.diagram?.title || "Diagram"}
      >
        {resolved.mode === "invalid" ? (
          // Same reasoning as the Mermaid failure path: say that it
          // didn't work, don't paste malformed JSON into the lesson.
          // Copy in the toolbar still retrieves it.
          <p className="flow-frame-note" role="status">
            This diagram could not be drawn.
          </p>
        ) : resolved.mode === "cycle" ? (
          <CycleDiagram diagram={resolved.diagram} />
        ) : resolved.mode === "timeline" ? (
          <TimelineDiagram diagram={resolved.diagram} />
        ) : resolved.mode === "linear" ? (
          <CardFlowchart diagram={resolved.diagram} />
        ) : (
          <MermaidDiagram rawCode={resolved.code} />
        )}
      </div>
    </div>
  );
}
