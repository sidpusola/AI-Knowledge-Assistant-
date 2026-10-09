import { getIcon } from "../lib/iconRegistry.js";

/**
 * Events in time, drawn against a spine.
 *
 * Distinct from the linear card list on purpose: a timeline says these
 * things happened in this order at these points, where a card chain
 * says each step causes the next. Project phases and history read as
 * the former; an algorithm reads as the latter.
 *
 * Only rendered for a diagram resolveDiagramType() calls a timeline, so
 * the nodes form one chain and can be walked in order.
 */
export default function TimelineDiagram({ diagram }) {
  const { title, nodes } = diagram;

  return (
    <figure className="flow-timeline">
      {title && <figcaption className="flow-timeline-title">{title}</figcaption>}

      <ol className="flow-timeline-list">
        {nodes.map((node) => {
          const Icon = getIcon(node.icon);
          return (
            <li key={node.id} className={`flow-timeline-item flow-card-${node.role}`}>
              <span className="flow-timeline-marker" aria-hidden="true">
                {Icon ? <Icon size={13} strokeWidth={2.2} /> : <span className="flow-timeline-dot" />}
              </span>
              <div className="flow-timeline-body">
                <span className="flow-timeline-label">{node.label}</span>
                {node.description && (
                  <span className="flow-timeline-description">{node.description}</span>
                )}
              </div>
            </li>
          );
        })}
      </ol>
    </figure>
  );
}
