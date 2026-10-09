import { getIcon } from "../lib/iconRegistry.js";

/**
 * A repeating process drawn as a ring.
 *
 * A cycle shown as a column of cards with a "repeats" footnote asks the
 * reader to imagine the loop. Drawing it round says it directly, which
 * is the whole point for a life cycle or the phases of the menstrual
 * cycle.
 *
 * The stages are HTML positioned over an SVG arc layer, rather than SVG
 * text. Mermaid's clipped labels on this project all came from SVG text
 * being measured in one font and drawn in another; HTML labels wrap and
 * size themselves, and CSS can theme them like everything else.
 *
 * Only rendered for a diagram resolveDiagramType() calls a cycle, so the
 * nodes are known to form one chain and can be walked in order.
 */

// Chip width in px. The ring is then sized around it rather than the
// other way round - a fixed-size ring either overlaps its chips at six
// stages or wastes half the frame at three.
const CHIP = 116;
// Centre-to-centre spacing between neighbouring chips. CHIP plus enough
// slack for an arrow to be visible in the gap; without the slack the
// arcs end up underneath the chips and the ring looks unconnected.
const SPACING = CHIP + 42;
const PAD = 10;
const MAX_SIZE = 460;

/**
 * Ring geometry for n stages.
 *
 * The radius comes from the chord between neighbours: chord =
 * 2R*sin(PI/n), so R = SPACING / (2*sin(PI/n)). The box then has to
 * hold that radius plus half a chip on each side.
 */
function geometryFor(n) {
  const needed = SPACING / (2 * Math.sin(Math.PI / Math.max(n, 2)));
  let radius = Math.max(96, needed);
  let size = 2 * (radius + CHIP / 2 + PAD);

  if (size > MAX_SIZE) {
    size = MAX_SIZE;
    radius = size / 2 - CHIP / 2 - PAD;
  }

  // How much of each arc to trim so it starts and ends clear of a chip
  // rather than vanishing beneath it.
  const clearance = Math.asin(Math.min(0.95, (CHIP / 2 + 8) / radius));
  return { size, radius, clearance, centre: size / 2 };
}

// Start at the top and run clockwise, the direction a cycle is read.
const angleFor = (index, total) => (index / total) * 2 * Math.PI - Math.PI / 2;

const pointAt = (index, total, g) => ({
  x: g.centre + g.radius * Math.cos(angleFor(index, total)),
  y: g.centre + g.radius * Math.sin(angleFor(index, total)),
});

/** Arc from one stage to the next, trimmed at both ends. Returns null
 *  when the stages sit too close for an arc to be worth drawing. */
function arcBetween(index, total, g) {
  const step = (2 * Math.PI) / total;
  const from = angleFor(index, total) + g.clearance;
  const to = angleFor(index, total) + step - g.clearance;
  if (to - from < 0.06) return null;

  const at = (angle) => [
    g.centre + g.radius * Math.cos(angle),
    g.centre + g.radius * Math.sin(angle),
  ];
  const [x1, y1] = at(from);
  const [x2, y2] = at(to);
  return `M ${x1.toFixed(1)} ${y1.toFixed(1)} A ${g.radius.toFixed(1)} ${g.radius.toFixed(1)} 0 0 1 ${x2.toFixed(1)} ${y2.toFixed(1)}`;
}

export default function CycleDiagram({ diagram }) {
  const { title, nodes } = diagram;
  const total = nodes.length;
  const g = geometryFor(total);

  return (
    <figure className="flow-cycle">
      {title && <figcaption className="flow-cycle-title">{title}</figcaption>}

      <div className="flow-cycle-stage" style={{ maxWidth: `${g.size}px` }}>
        <svg
          className="flow-cycle-arcs"
          viewBox={`0 0 ${g.size} ${g.size}`}
          preserveAspectRatio="xMidYMid meet"
          aria-hidden="true"
        >
          <defs>
            <marker
              id="flow-cycle-arrow"
              viewBox="0 0 10 10"
              refX="8"
              refY="5"
              markerWidth="5"
              markerHeight="5"
              orient="auto-start-reverse"
            >
              <path d="M 0 1 L 9 5 L 0 9 z" fill="currentColor" />
            </marker>
          </defs>
          {nodes.map((node, index) => {
            const d = arcBetween(index, total, g);
            return d ? (
              <path
                key={node.id}
                d={d}
                fill="none"
                stroke="currentColor"
                strokeWidth="1.6"
                markerEnd="url(#flow-cycle-arrow)"
              />
            ) : null;
          })}
        </svg>

        {/* An ordered list, so the stages are still a sequence to a
            screen reader - the ring is a visual arrangement only. */}
        <ol className="flow-cycle-nodes">
          {nodes.map((node, index) => {
            const { x, y } = pointAt(index, total, g);
            const Icon = getIcon(node.icon);
            return (
              <li
                key={node.id}
                className={`flow-cycle-node flow-card-${node.role}`}
                style={{
                  left: `${(x / g.size) * 100}%`,
                  top: `${(y / g.size) * 100}%`,
                }}
              >
                {Icon && (
                  <span className="flow-cycle-icon" aria-hidden="true">
                    <Icon size={14} strokeWidth={2} />
                  </span>
                )}
                <span className="flow-cycle-label">{node.label}</span>
                {node.description && (
                  <span className="flow-cycle-description">{node.description}</span>
                )}
              </li>
            );
          })}
        </ol>
      </div>
    </figure>
  );
}
