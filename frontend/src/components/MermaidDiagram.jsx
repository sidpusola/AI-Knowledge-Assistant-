import { useEffect, useMemo, useRef, useState } from "react";
import mermaid from "mermaid";
import {
  applyRoleStyles,
  preferHorizontal,
  repairMermaid,
  stripModelStyling,
} from "../lib/diagramLayout.js";

// Mermaid measures label text to size each box. "inherit" is not a font
// it can measure with, so it sized boxes against one font and drew them
// in another - which clipped longer labels ("Hatch into Larva" rendered
// as "Hatch into Lar"). The app's real stack has to be named here.
const FONT_STACK =
  '-apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';

// theme "base" + explicit variables, rather than the built-in "default"
// and "dark" themes: those bring their own palette and a raised, shadowed
// node style that fights the flat look of the rest of the UI. Everything
// below is driven by the same tokens as App.css.
const PALETTE = {
  light: {
    nodeFill: "#ffffff",
    nodeBorder: "#d9d0c4",
    text: "#2b2622",
    line: "#a89f94",
    clusterFill: "#faf7f2",
    clusterBorder: "#ece5da",
    // Same values as the card renderer's role tints in App.css, so a
    // Mermaid diagram and a card diagram read as one product.
    roles: {
      startBg: "#eef7ef", startBorder: "#cde2d1",
      endBg: "#eff2fb", endBorder: "#d5ddf1",
      decisionBg: "#fdf4e3", decisionBorder: "#eddcb9",
      text: "#2b2622",
    },
  },
  dark: {
    nodeFill: "#1b1b1b",
    nodeBorder: "#3a3a3a",
    text: "#f3f4f6",
    line: "#6b7280",
    clusterFill: "#111111",
    clusterBorder: "#2a2a2a",
    roles: {
      startBg: "#15231a", startBorder: "#2b4030",
      endBg: "#171c2b", endBorder: "#2e3755",
      decisionBg: "#272013", decisionBorder: "#463a1e",
      text: "#f3f4f6",
    },
  },
};

function configFor(themeName) {
  const c = PALETTE[themeName === "dark" ? "dark" : "light"];
  return {
    startOnLoad: false,
    // Keeps mermaid from injecting its own red error box into the DOM when
    // a diagram doesn't parse. That failure is handled here - by falling
    // back to the source text - not splashed into a chat bubble.
    suppressErrorRendering: true,
    securityLevel: "strict",
    fontFamily: FONT_STACK,
    theme: "base",
    themeVariables: {
      fontFamily: FONT_STACK,
      fontSize: "14px",
      primaryColor: c.nodeFill,
      primaryBorderColor: c.nodeBorder,
      primaryTextColor: c.text,
      secondaryColor: c.clusterFill,
      tertiaryColor: c.clusterFill,
      background: "transparent",
      mainBkg: c.nodeFill,
      nodeBorder: c.nodeBorder,
      lineColor: c.line,
      textColor: c.text,
      clusterBkg: c.clusterFill,
      clusterBorder: c.clusterBorder,
      edgeLabelBackground: c.clusterFill,
      titleColor: c.text,
    },
    flowchart: {
      // Straight connectors read as a cleaner diagram than mermaid's
      // default curves, and tighter spacing keeps a ten-step process from
      // becoming a page-long column.
      curve: "linear",
      nodeSpacing: 28,
      rankSpacing: 34,
      padding: 10,
      useMaxWidth: true,
      diagramPadding: 8,
      // Wrap long labels onto a second line instead of growing one very
      // wide box. App.css pins the rendered font to the size used for
      // this measurement - if either changes, change both.
      wrappingWidth: 170,
    },
  };
}

mermaid.initialize(configFor("light"));

/** Track the app's light/dark setting, which App.jsx writes to
 *  <html data-theme>. Mermaid bakes colours into the SVG at render time,
 *  so a diagram drawn in light mode stays light after the toggle unless
 *  it is redrawn - hence observing the attribute rather than reading it
 *  once. */
function useThemeName() {
  const [theme, setTheme] = useState(
    () => document.documentElement.dataset.theme || "light"
  );

  useEffect(() => {
    const observer = new MutationObserver(() =>
      setTheme(document.documentElement.dataset.theme || "light")
    );
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["data-theme"],
    });
    return () => observer.disconnect();
  }, []);

  return theme;
}

let diagramCounter = 0;

// Rendered SVG keyed by source + theme, shared across every instance.
// Drawing is async, so without this a remount blanks the diagram and
// redraws it - visible as a flash. Read during render, not in an effect,
// so a cached diagram paints on the very first frame.
const svgCache = new Map();

const keyFor = (theme, code) => `${theme}\u0000${code}`;

export default function MermaidDiagram({ rawCode }) {
  // Direction is decided here rather than trusted from the model -
  // see lib/diagramLayout.js. Done before the cache key is built so
  // the cached SVG and the key always describe the same diagram.
  const theme = useThemeName();
  // Order matters: drop the model's own styling (it is often invalid),
  // put back a missing `flowchart TD` header, choose the direction from
  // the repaired graph, then colour it by role.
  const code = useMemo(
    () =>
      applyRoleStyles(
        preferHorizontal(repairMermaid(stripModelStyling(rawCode))),
        PALETTE[theme === "dark" ? "dark" : "light"].roles
      ),
    [rawCode, theme]
  );
  const cacheKey = keyFor(theme, code);
  const cached = svgCache.get(cacheKey);

  // One piece of state tagged with the key it belongs to, so a result for
  // a previous code/theme is never shown against the current one.
  const [result, setResult] = useState({ key: null, svg: "", failed: false });
  // Guards against a slow render landing after the inputs changed again.
  const renderToken = useRef(0);

  useEffect(() => {
    if (svgCache.has(cacheKey)) return undefined;

    const token = ++renderToken.current;
    let cancelled = false;

    (async () => {
      try {
        mermaid.initialize(configFor(theme));

        // parse() first, so invalid syntax is a caught rejection rather
        // than a half-drawn diagram.
        await mermaid.parse(code);
        const { svg } = await mermaid.render(`mermaid-${diagramCounter++}`, code);

        svgCache.set(cacheKey, svg);
        if (!cancelled && token === renderToken.current) {
          setResult({ key: cacheKey, svg, failed: false });
        }
      } catch {
        // A local model will sometimes emit almost-valid Mermaid. Showing
        // the source beats showing nothing: the explanation around it is
        // still useful, and the student can see what was attempted.
        if (!cancelled && token === renderToken.current) {
          setResult({ key: cacheKey, svg: "", failed: true });
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [cacheKey, code, theme]);

  const current = result.key === cacheKey ? result : null;
  const svg = cached || current?.svg || "";
  const failed = !svg && Boolean(current?.failed);

  if (failed) {
    // The source is deliberately not printed. A student has no use for
    // broken Mermaid, and the surrounding explanation usually carries
    // the same content in words. The frame's copy button still yields
    // the source for anyone debugging.
    return (
      <p className="chat-diagram-note" role="status">
        This diagram could not be drawn.
      </p>
    );
  }

  if (!svg) return <div className="chat-diagram chat-diagram-pending" />;

  // mermaid.render() returns SVG it generated itself from text it already
  // parsed, with securityLevel "strict" (no inline scripts, HTML labels
  // escaped), so this is not arbitrary model output reaching the DOM.
  return <div className="chat-diagram" dangerouslySetInnerHTML={{ __html: svg }} />;
}
