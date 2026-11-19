import {
  BarChart3, BookOpen, Briefcase, Compass, FileText, GraduationCap,
  Layers, ListChecks, MessageCircle, PanelLeftClose,
} from "lucide-react";

import UserMenu from "./UserMenu.jsx";

const NAV = [
  { id: "chat", label: "Ask", Icon: MessageCircle, group: "Study" },
  { id: "course", label: "My course", Icon: BookOpen, group: "Study" },
  { id: "guide", label: "Guide", Icon: Compass, group: "Study" },
  { id: "documents", label: "Documents", Icon: FileText, group: "Study" },
  { id: "quiz", label: "Quiz", Icon: ListChecks, group: "Practice" },
  { id: "cards", label: "Flashcards", Icon: Layers, group: "Practice" },
  { id: "progress", label: "Progress", Icon: BarChart3, group: "Your track" },
  { id: "placement", label: "Placement", Icon: Briefcase, group: "Your track" },
];

const GROUPS = ["Study", "Practice", "Your track"];

function NavItem({ item, active, onSelect }) {
  const { id, label, Icon } = item;
  return (
    <button
      type="button"
      className={`nav-item ${active ? "on" : ""}`}
      onClick={() => onSelect(id)}
      aria-current={active ? "page" : undefined}
      // The visible label is hidden when the rail is a strip of icons.
      aria-label={label}
      title={label}
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
          {GROUPS.map((group) => (
            <div className="nav-group" key={group}>
              <div className="nav-head">{group}</div>
              {NAV.filter((item) => item.group === group).map((item) => (
                <NavItem
                  key={item.id}
                  item={item}
                  active={activeTab === item.id}
                  onSelect={onTabChange}
                />
              ))}
            </div>
          ))}
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
