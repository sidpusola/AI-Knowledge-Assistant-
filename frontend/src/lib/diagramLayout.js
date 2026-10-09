/**
 * Pick a sensible flow direction for a Mermaid flowchart.
 *
 * The model is asked to choose LR for straight sequences and TD for
 * branching ones, and it ignores that instruction - measured 7/7
 * top-down across linear and branching questions alike. qwen2.5 has a
 * very strong prior for `flowchart TD`. Rather than keep adding prompt
 * text it doesn't follow, the decision is made here, where it is
 * deterministic: the same reasoning the prompt describes, applied to the
 * diagram that actually came back.
 *
 * A straight chain rendered top-down is a tall narrow ribbon that wastes
 * the width of the bubble and forces scrolling; the same chain
 * left-to-right fits on screen. Branching diagrams keep TD, which is
 * where the vertical room is genuinely needed.
 */

const DIRECTION_RE = /^(\s*)(flowchart|graph)\s+(TD|TB)\b/m;

const NOT_NODE_IDS = ["subgraph", "end", "flowchart", "graph", "TD", "TB", "LR", "RL"];

/** Edges, with node labels stripped first - otherwise an edge whose
 *  source carries an inline label (A["x"] --> B) is invisible. */
function edgesOf(code) {
  const found = [];
  for (const raw of code.split("\n")) {
    const line = raw
      .replace(/\{\{.*?\}\}/g, " ")
      .replace(/\(\[.*?\]\)/g, " ")
      .replace(/\[.*?\]/g, " ")
      .replace(/\(.*?\)/g, " ")
      .replace(/\{.*?\}/g, " ")
      .replace(/\|[^|]*\|/g, " ")
      // the other edge-label form: A -- Yes --> B. Must be handled before
      // the split, or "Yes" is read as a node and the real source is lost.
      .replace(/--+\s*[^->|]+?\s*--+>/g, " --> ");

    const ids = line
      .split(/-{2,}>|={2,}>|-\.->/)
      .map((part) =>
        (part.match(/\b[A-Za-z]\w*\b/g) || []).filter((id) => !NOT_NODE_IDS.includes(id))
      );

    for (let i = 0; i < ids.length - 1; i++) {
      const from = ids[i];
      const to = ids[i + 1];
      if (from.length && to.length) found.push([from[from.length - 1], to[0]]);
    }
  }
  return found;
}

export function preferHorizontal(code) {
  if (typeof code !== "string" || !DIRECTION_RE.test(code)) return code;

  const edges = edgesOf(code);
  // Too small to matter - two or three boxes look fine either way.
  if (edges.length < 3) return code;

  const outDegree = new Map();
  for (const [from] of edges) outDegree.set(from, (outDegree.get(from) || 0) + 1);
  const forks = [...outDegree.values()].filter((n) => n > 1).length;

  // More than one split and the layout genuinely needs vertical room.
  if (forks > 1) return code;

  return code.replace(DIRECTION_RE, (_match, indent, keyword) => `${indent}${keyword} LR`);
}

// Every Mermaid diagram must open by declaring its type. These are the
// keywords that count as a declaration.
const DIAGRAM_KEYWORDS = new Set([
  "flowchart", "graph", "sequencediagram", "classdiagram", "statediagram",
  "statediagram-v2", "erdiagram", "journey", "gantt", "pie", "quadrantchart",
  "requirementdiagram", "gitgraph", "mindmap", "timeline", "sankey-beta",
  "xychart-beta", "block-beta", "packet-beta", "kanban", "architecture-beta",
  "radar-beta", "treemap-beta", "c4context", "zenuml",
]);

const EDGE_RE = /(-{2,}>|={2,}>|-\.->|-{3,}|:::)/;

/**
 * Put back a missing diagram-type header.
 *
 * The model sometimes emits the body of a flowchart with no opening
 * line - just `A[Start] --> B[Hatch]` and so on. Mermaid cannot parse
 * that and the whole diagram is lost, even though everything after the
 * first line was fine. The prompt already says to start with
 * `flowchart LR`; this is the deterministic backstop for when it
 * doesn't, in the same spirit as preferHorizontal.
 *
 * Only ever ADDS a header, and only when the content already looks like
 * flowchart edges - so a diagram that declares itself, or a block that
 * is something else entirely, is returned untouched.
 */
export function repairMermaid(code) {
  if (typeof code !== "string") return code;

  const trimmed = code.replace(/^\s+/, "");
  const firstLine = trimmed.split("\n", 1)[0].trim();
  const firstWord = firstLine.split(/[\s{]/, 1)[0].toLowerCase();

  if (DIAGRAM_KEYWORDS.has(firstWord)) return code;
  if (!EDGE_RE.test(trimmed)) return code;

  // TD rather than LR because preferHorizontal runs next and decides
  // the direction properly from the shape of the graph.
  return `flowchart TD\n${trimmed}`;
}

/**
 * Remove any styling the model wrote for itself.
 *
 * Asked for a colourful diagram, qwen2.5 invents syntax - it produced
 * `C[Power light turns on]{ style C fill:#99ff99 }`, which is not valid
 * Mermaid and loses the whole diagram. Colour is applied by the app
 * instead (see applyRoleStyles), so anything the model wrote here is
 * both unnecessary and a parse risk.
 */
export function stripModelStyling(code) {
  if (typeof code !== "string") return code;

  return code
    // a brace group holding style/fill directives, attached to a node
    .replace(/\{[^{}]*(?:style|fill|stroke)\s*[:A-Za-z][^{}]*\}/g, "")
    // whole-line style / classDef / class directives
    .replace(/^[ \t]*(?:style|classDef|class)\s+[^\n]*$/gm, "")
    // the blank lines that leaves behind
    .replace(/\n{3,}/g, "\n\n")
    .trimEnd();
}

/**
 * Colour the diagram by the role each node plays.
 *
 * Every flowchart otherwise renders as one undifferentiated set of grey
 * boxes, so a life cycle and a troubleshooting tree look identical. The
 * entry point, the exits and the decisions are the three things worth
 * telling apart at a glance, and they can be derived from the graph
 * rather than asked for: a node with no incoming edge starts it, one
 * with no outgoing edge ends it, and a {{...}} node is a decision.
 *
 * Colours are passed in so they can follow the light/dark theme, and
 * they match the card renderer's palette - the two modes should look
 * like one product.
 */
export function applyRoleStyles(code, palette) {
  if (typeof code !== "string" || !palette) return code;
  if (!/^\s*(flowchart|graph)\b/m.test(code)) return code;

  const edges = edgesOf(code);
  if (!edges.length) return code;

  const sources = new Set(edges.map(([from]) => from));
  const targets = new Set(edges.map(([, to]) => to));
  const declared = new Set();
  for (const match of code.matchAll(/^\s*([A-Za-z]\w*)\s*[[({]/gm)) {
    declared.add(match[1]);
  }
  for (const id of [...sources, ...targets]) declared.add(id);

  const decisions = new Set();
  for (const match of code.matchAll(/\b([A-Za-z]\w*)\s*\{/g)) decisions.add(match[1]);

  const starts = [...declared].filter((id) => !targets.has(id) && sources.has(id));
  const ends = [...declared].filter((id) => !sources.has(id) && targets.has(id));
  const decisionIds = [...decisions].filter((id) => declared.has(id));

  const lines = [];
  const define = (name, bg, border) =>
    `    classDef ${name} fill:${bg},stroke:${border},color:${palette.text},stroke-width:1.2px;`;

  if (starts.length) {
    lines.push(define("roleStart", palette.startBg, palette.startBorder));
    lines.push(`    class ${starts.join(",")} roleStart;`);
  }
  if (ends.length) {
    lines.push(define("roleEnd", palette.endBg, palette.endBorder));
    lines.push(`    class ${ends.join(",")} roleEnd;`);
  }
  if (decisionIds.length) {
    lines.push(define("roleDecision", palette.decisionBg, palette.decisionBorder));
    lines.push(`    class ${decisionIds.join(",")} roleDecision;`);
  }

  return lines.length ? `${code}\n${lines.join("\n")}` : code;
}
