import {
  BarChart3, BookOpen, Briefcase, Compass, FileText, GraduationCap,
  Layers, ListChecks, MessageCircle, PanelLeftClose,
} from "lucide-react";

import UserMenu from "./UserMenu.jsx";

/* The two screens that have a backend behind them. Everything else in
 * the design is a feature that does not exist yet - see BLOCKED below. */
const LIVE_NAV = [
  { id: "chat", label: "Ask", Icon: MessageCircle, group: "Study" },
  { id: "documents", label: "Documents", Icon: FileText, group: "Study" },
];

/* Shown, dimmed and inert. Hiding them would lose the shape of the
 * product the design describes; wiring them to empty screens would
 * imply they work. Dimmed with a "soon" marker is the honest middle. */
const PLANNED_NAV = [
  { id: "course", label: "My course", Icon: BookOpen, group: "Study" },
  { id: "guide", label: "Guide", Icon: Compass, group: "Study" },
  { id: "quiz", label: "Quiz", Icon: ListChecks, group: "Practice" },
  { id: "cards", label: "Flashcards", Icon: Layers, group: "Practice" },
  { id: "progress", label: "Progress", Icon: BarChart3, group: "Your track" },
  { id: "placement", label: "Placement", Icon: Briefcase, group: "Your track" },
];

const GROUPS = ["Study", "Practice", "Your track"];

function NavItem({ item, active, onSelect }) {
  const { id, label, Icon } = item;
  if (!onSelect) {
    return (
      <div className="nav-item soon" aria-disabled="true">
        <Icon aria-hidden="true" />
        <span>{label}</span>
        <span className="soon-tag">soon</span>
      </div>
    );
  }
  return (
    <button
      type="button"
      className={`nav-item ${active ? "on" : ""}`}
      onClick={() => onSelect(id)}
      aria-current={active ? "page" : undefined}
    >
      <Icon aria-hidden="true" />
      <span>{label}</span>
    </button>
  );
}

export default function AppShell({
  activeTab,
  onTabChange,
  theme,
  onToggleTheme,
  user,
  authConfigured,
  onUserChange,
  railCollapsed,
  onToggleRail,
  children,
}) {
  return (
    <div className={`app ${railCollapsed ? "rail-collapsed" : ""}`}>
      <aside className="rail">
        <div className="brand">
          {railCollapsed ? (
            // Collapsed, the logo is the control. A 64px strip has room
            // for one thing, and a student who has collapsed the rail
            // needs the way back more than they need the mark.
            <button
              type="button"
              className="brand-mark is-button"
              onClick={onToggleRail}
              aria-label="Expand the sidebar"
              title="Expand the sidebar"
              aria-expanded={false}
            >
              <GraduationCap aria-hidden="true" />
            </button>
          ) : (
            <>
              <div className="brand-mark"><GraduationCap aria-hidden="true" /></div>
              <span className="brand-name">Anudip</span>
              <button
                type="button"
                className="rail-toggle"
                onClick={onToggleRail}
                aria-label="Collapse the sidebar"
                title="Collapse the sidebar"
                aria-expanded
              >
                <PanelLeftClose aria-hidden="true" />
              </button>
            </>
          )}
        </div>

        <nav>
          {GROUPS.map((group) => {
            const live = LIVE_NAV.filter((item) => item.group === group);
            const planned = PLANNED_NAV.filter((item) => item.group === group);
            if (!live.length && !planned.length) return null;
            return (
              <div className="nav-group" key={group}>
                <div className="nav-head">{group}</div>
                {live.map((item) => (
                  <NavItem
                    key={item.id}
                    item={item}
                    active={activeTab === item.id}
                    onSelect={onTabChange}
                  />
                ))}
                {planned.map((item) => (
                  <NavItem key={item.id} item={item} />
                ))}
              </div>
            );
          })}
        </nav>

        <div className="rail-footer">
          <UserMenu
            user={user}
            configured={authConfigured}
            onChange={onUserChange}
            theme={theme}
            onToggleTheme={onToggleTheme}
            dropUp
          />
        </div>

      </aside>

      <div className="main">
        {children}
      </div>
    </div>
  );
}
