import { ArrowDown, ArrowRight, RefreshCw } from "lucide-react";

import { getIcon } from "../lib/iconRegistry.js";
import { cycleLabel, hasCycleBack } from "../lib/diagramSchema.js";

/**
 * Mode A: a simple sequence drawn as cards rather than boxes-and-arrows.
 *
 * Only ever rendered for a diagram isLinearChain() has approved, so the
 * nodes can be walked in order - there is no layout to compute, which is
 * exactly why this looks tidier than Mermaid for a life cycle. Anything
 * that branches goes to Mermaid instead; see FlowchartRenderer.
 *
 * Marked up as an ordered list so the stages are a sequence to a screen
 * reader too, not a pile of divs. Connectors are aria-hidden: the list
 * order already carries that meaning, and "arrow down" read aloud
 * between every stage is noise.
 */
export default function CardFlowchart({ diagram }) {
  const { title, nodes, layout } = diagram;
  const repeats = hasCycleBack(diagram);
  const repeatLabel = cycleLabel(diagram) || "Cycle repeats";
  const Connector = layout === "horizontal" ? ArrowRight : ArrowDown;

  return (
    <figure className={`flow-cards flow-cards-${layout}`}>
      {title && <figcaption className="flow-cards-title">{title}</figcaption>}

      <ol className="flow-cards-list">
        {nodes.map((node, index) => {
          const Icon = getIcon(node.icon);
          return (
            <li key={node.id} className="flow-cards-item">
              <div className={`flow-card flow-card-${node.role}`}>
                {Icon && (
                  <span className="flow-card-icon" aria-hidden="true">
                    <Icon size={17} strokeWidth={2} />
                  </span>
                )}
                <div className="flow-card-body">
                  <span className="flow-card-label">{node.label}</span>
                  {node.description && (
                    <span className="flow-card-description">{node.description}</span>
                  )}
                </div>
              </div>

              {index < nodes.length - 1 && (
                <span className="flow-cards-connector" aria-hidden="true">
                  <Connector size={15} strokeWidth={2} />
                </span>
              )}
            </li>
          );
        })}
      </ol>

      {repeats && (
        <p className="flow-cards-repeat">
          <RefreshCw size={13} strokeWidth={2} aria-hidden="true" />
          {repeatLabel}
        </p>
      )}
    </figure>
  );
}
