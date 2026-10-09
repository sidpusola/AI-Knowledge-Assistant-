/**
 * Validate and normalise the structured-diagram JSON a model may return.
 *
 * Everything here is untrusted input. Nothing is rendered unless it has
 * been through parseFlowchart, which checks shape and types, clamps every
 * string to a sane length, drops unknown icons, and rejects edges that
 * point at nodes which don't exist. A malformed diagram returns a reason
 * instead of throwing, so the caller can fall back to showing the source.
 */
import { getIcon } from "./iconRegistry.js";

const LIMITS = {
  title: 80,
  label: 60,
  description: 140,
  edgeLabel: 40,
  nodes: 24,
  edges: 48,
};

const ROLES = new Set(["start", "process", "decision", "end"]);

/**
 * What the diagram is, conceptually. The renderer is chosen from this
 * rather than from the question wording, so "menstrual cycle" gets a
 * ring and "types of machine learning" gets a tree.
 *
 *   linear        a sequence that ends          -> cards
 *   cycle         a sequence that repeats       -> ring
 *   timeline      events ordered by time        -> timeline
 *   decision_tree conditional logic             -> Mermaid
 *   hierarchy     parent/child classification   -> Mermaid
 *   parallel      concurrent work that rejoins  -> Mermaid
 */
const DIAGRAM_TYPES = new Set([
  "linear", "cycle", "timeline", "decision_tree", "hierarchy", "parallel",
]);

/** Visual treatments. The model picks one; unknown values fall back to
 *  minimal rather than being rejected - a theme is never load-bearing. */
const THEMES = new Set(["minimal", "biology", "technical", "education"]);
// A timeline naturally keys its stages by year ("1956"), so ids may
// start with a digit. They also become Mermaid identifiers in the
// fallback, where a leading digit is not safe - mermaidId() prefixes
// those rather than rejecting a perfectly sensible diagram.
const ID_RE = /^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$/;
const mermaidId = (id) => (/^[0-9]/.test(id) ? `n${id}` : id);

const clamp = (value, max) =>
  typeof value === "string" ? value.trim().slice(0, max) : "";

function fail(reason) {
  return { ok: false, reason };
}

export function parseFlowchart(text) {
  let raw;
  try {
    raw = JSON.parse(text);
  } catch {
    return fail("not valid JSON");
  }

  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return fail("not a JSON object");
  }
  // "diagram_type" is the current field; "type": "flowchart" was the
  // first version of this format and still turns up, so both are read.
  const declaredType = typeof raw.diagram_type === "string" ? raw.diagram_type.trim() : "";
  const legacyType = typeof raw.type === "string" ? raw.type.trim() : "";
  if (!declaredType && legacyType !== "flowchart") {
    return fail(`unexpected type: ${String(raw.type ?? raw.diagram_type).slice(0, 20)}`);
  }
  if (!Array.isArray(raw.nodes) || raw.nodes.length === 0) {
    return fail("no nodes");
  }
  if (raw.nodes.length > LIMITS.nodes) {
    return fail(`too many nodes (${raw.nodes.length})`);
  }

  const seen = new Set();
  const nodes = [];
  for (const node of raw.nodes) {
    if (!node || typeof node !== "object") return fail("malformed node");
    const id = typeof node.id === "string" ? node.id.trim() : "";
    if (!ID_RE.test(id)) return fail(`bad node id: ${id.slice(0, 20) || "(empty)"}`);
    if (seen.has(id)) return fail(`duplicate node id: ${id}`);
    seen.add(id);

    const label = clamp(node.label, LIMITS.label);
    if (!label) return fail(`node "${id}" has no label`);

    nodes.push({
      id,
      label,
      description: clamp(node.description, LIMITS.description) || null,
      // Unknown icon names are dropped rather than rejected - the diagram
      // is still perfectly readable without one.
      icon: getIcon(node.icon) ? node.icon : null,
      role: ROLES.has(node.role) ? node.role : "process",
      // Remembered so the start/end defaulting below can tell "the model
      // said process" from "the model said nothing".
      roleGiven: ROLES.has(node.role),
    });
  }

  const rawEdges = Array.isArray(raw.edges) ? raw.edges : [];
  if (rawEdges.length > LIMITS.edges) return fail("too many edges");

  const edges = [];
  for (const edge of rawEdges) {
    if (!edge || typeof edge !== "object") return fail("malformed edge");
    // from/to and source/target are both accepted - the model uses
    // whichever it saw last, and arguing with it costs more than this.
    const rawFrom = edge.from ?? edge.source;
    const rawTo = edge.to ?? edge.target;
    const from = typeof rawFrom === "string" ? rawFrom.trim() : "";
    const to = typeof rawTo === "string" ? rawTo.trim() : "";
    if (!seen.has(from) || !seen.has(to)) {
      return fail(`edge points at an unknown node (${from || "?"} -> ${to || "?"})`);
    }
    edges.push({ from, to, label: clamp(edge.label, LIMITS.edgeLabel) || null });
  }

  // The model often leaves every node as "process". The first and last
  // stage of a sequence read better when they're visually distinct, so
  // they're filled in here - but only when it didn't mark any itself,
  // so an explicit choice is never overridden.
  const last = nodes.length - 1;
  if (nodes.length > 1 && !nodes.some((n) => n.role === "start") && !nodes[0].roleGiven) {
    nodes[0].role = "start";
  }
  if (nodes.length > 1 && !nodes.some((n) => n.role === "end") && !nodes[last].roleGiven) {
    nodes[last].role = "end";
  }
  for (const node of nodes) delete node.roleGiven;

  return {
    ok: true,
    diagram: {
      title: clamp(raw.title, LIMITS.title) || null,
      layout: raw.layout === "horizontal" ? "horizontal" : "vertical",
      // A declared type is trusted only if it's one we render; anything
      // else is worked out from the graph's own shape below.
      diagramType: DIAGRAM_TYPES.has(declaredType) ? declaredType : null,
      theme: THEMES.has(raw.theme) ? raw.theme : "minimal",
      nodes,
      edges,
    },
  };
}

/**
 * Can this be drawn as a single column of cards?
 *
 * Only when it is one chain: every node leads to at most one other, and
 * nothing is joined from two directions. One edge back to the first node
 * is allowed and treated as "the cycle repeats" - that is the shape of a
 * life cycle, which is the main thing card mode exists for. Anything
 * else is a graph and belongs in Mermaid.
 */
export function isLinearChain({ nodes, edges }) {
  if (nodes.length < 2) return false;

  const firstId = nodes[0].id;
  const forward = edges.filter((e) => !(e.to === firstId && e.from !== firstId));
  const cycleEdges = edges.length - forward.length;
  if (cycleEdges > 1) return false;

  const out = new Map();
  const into = new Map();
  for (const { from, to } of forward) {
    if (from === to) return false;
    out.set(from, (out.get(from) || 0) + 1);
    into.set(to, (into.get(to) || 0) + 1);
  }
  if ([...out.values()].some((n) => n > 1)) return false;
  if ([...into.values()].some((n) => n > 1)) return false;

  // Every node must sit on the chain, in the order given.
  for (let i = 0; i < nodes.length - 1; i++) {
    const hasNext = forward.some(
      (e) => e.from === nodes[i].id && e.to === nodes[i + 1].id
    );
    if (!hasNext) return false;
  }
  return true;
}

/** Fallback for a valid diagram that isn't a simple chain: hand it to
 *  Mermaid rather than forcing it into a column of cards. */
export function toMermaid({ nodes, edges }) {
  const escape = (s) => s.replace(/"/g, "'");
  const shape = (node) => {
    const id = mermaidId(node.id);
    return node.role === "decision"
      ? `${id}{{"${escape(node.label)}"}}`
      : node.role === "start" || node.role === "end"
        ? `${id}(["${escape(node.label)}"])`
        : `${id}["${escape(node.label)}"]`;
  };

  const lines = ["flowchart TD", ...nodes.map((n) => `    ${shape(n)}`)];
  for (const edge of edges) {
    const from = mermaidId(edge.from);
    const to = mermaidId(edge.to);
    lines.push(
      edge.label
        ? `    ${from} -->|"${escape(edge.label)}"| ${to}`
        : `    ${from} --> ${to}`
    );
  }
  return lines.join("\n");
}

/** Does the whole chain have a cycle back to its first node? */
export function hasCycleBack({ nodes, edges }) {
  if (!nodes.length) return false;
  const firstId = nodes[0].id;
  return edges.some((e) => e.to === firstId && e.from !== firstId);
}

export function cycleLabel({ nodes, edges }) {
  if (!nodes.length) return null;
  const firstId = nodes[0].id;
  const edge = edges.find((e) => e.to === firstId && e.from !== firstId);
  return edge ? edge.label : null;
}

/**
 * Which renderer a validated diagram should use.
 *
 * The model's declared diagram_type is a hint, not an instruction: it
 * will happily label a branching graph "cycle". A declared type is only
 * honoured when the graph can actually be drawn that way - cards, rings
 * and timelines can each express exactly one chain - and otherwise the
 * diagram goes to Mermaid, which can lay out anything.
 *
 * With no declared type, the shape decides: a chain that returns to its
 * first node is a cycle, a chain that doesn't is linear.
 */
// Above this many stages the ring's chips start to collide and its
// arrows shrink to nothing (measured in CycleDiagram's geometry: 6 is
// the last count where both still work). A long cycle is drawn as a
// card chain instead, which still shows that it repeats.
const MAX_RING_STAGES = 6;

export function resolveDiagramType(diagram) {
  const chain = isLinearChain(diagram);
  const cyclic = hasCycleBack(diagram);
  const declared = diagram.diagramType;
  const ringFits = diagram.nodes.length <= MAX_RING_STAGES;

  // These three are chain-only renderers.
  if (declared === "cycle") {
    if (!chain || !cyclic) return "mermaid";
    return ringFits ? "cycle" : "linear";
  }
  if (declared === "timeline") return chain ? "timeline" : "mermaid";
  if (declared === "linear") {
    if (!chain) return "mermaid";
    return cyclic && ringFits ? "cycle" : "linear";
  }

  // decision_tree, hierarchy and parallel are all graphs - Mermaid's job.
  if (declared) return "mermaid";

  if (!chain) return "mermaid";
  return cyclic && ringFits ? "cycle" : "linear";
}
