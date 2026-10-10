import { useEffect, useId, useRef, useState } from "react";
import { LogIn, LogOut, Moon, Sun, User, UserCircle2 } from "lucide-react";

import { googleLoginUrl, logout, saveProfile } from "../api.js";

/** Initials for the avatar: "Asha Rao" -> AR, otherwise the first letter
 *  of the email. Never empty, so the circle is never blank. */
function initialsFor(user) {
  const name = (user.display_name || "").trim();
  if (name) {
    const parts = name.split(/\s+/);
    return ((parts[0][0] || "") + (parts.length > 1 ? parts[parts.length - 1][0] : ""))
      .toUpperCase();
  }
  return (user.email || "?")[0].toUpperCase();
}

/**
 * The account control in the header.
 *
 * One circular button either way - initials when signed in, a person
 * glyph when not. No "Sign in" label: the header has a fixed amount of
 * room and a word there pushes the layout around every time auth state
 * resolves.
 *
 * The menu exists signed out as well as in, because it now holds the
 * light/dark switch. Hiding that behind a sign-in would mean a student
 * on a shared machine cannot turn the lights down until they log in.
 */
export default function UserMenu({
  user, configured, onChange, theme, onToggleTheme, dropUp = false,
}) {
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState(false);
  const containerRef = useRef(null);
  const menuId = useId();

  const close = () => {
    setOpen(false);
    setEditing(false);
  };

  // Close on a click anywhere else, or on Escape. Without both, the menu
  // is a trap for anyone not using a mouse.
  useEffect(() => {
    if (!open) return undefined;

    const onPointerDown = (event) => {
      if (!containerRef.current?.contains(event.target)) close();
    };
    const onKeyDown = (event) => {
      if (event.key === "Escape") close();
    };

    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open]);

  const handleSignOut = async () => {
    await logout();
    onChange(null);
    close();
  };

  const label = user
    ? `Account: ${user.display_name || user.email}`
    : "Account and appearance";

  return (
    <div className={`user-menu ${dropUp ? "drop-up" : ""}`} ref={containerRef}>
      <button
        type="button"
        className={`user-avatar ${user ? "" : "is-anon"}`}
        onClick={() => setOpen((value) => !value)}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        // The visible initials alone read as two stray letters, so the
        // accessible name carries who is signed in.
        aria-label={label}
        title={label}
      >
        {user ? initialsFor(user) : <User aria-hidden="true" />}
      </button>

      {open && (
        <div className="user-dropdown" id={menuId} role="menu">
          <div className="user-dropdown-identity">
            {user ? (
              <>
                <span className="user-dropdown-name">
                  {user.display_name || "No name set"}
                </span>
                <span className="user-dropdown-email">{user.email}</span>
                {!user.onboarded && (
                  <span className="user-dropdown-hint">Finish your profile</span>
                )}
              </>
            ) : (
              <>
                <span className="user-dropdown-name">Not signed in</span>
                <span className="user-dropdown-email">
                  {configured
                    ? "Sign in to keep your conversations."
                    : "Sign-in is not configured on this server."}
                </span>
              </>
            )}
          </div>

          {editing ? (
            <ProfileForm
              user={user}
              onSaved={(updated) => {
                onChange(updated);
                setEditing(false);
              }}
              onCancel={() => setEditing(false)}
            />
          ) : (
            <>
              <button
                type="button"
                className="user-dropdown-item"
                role="menuitem"
                onClick={onToggleTheme}
              >
                {theme === "dark"
                  ? <Sun size={14} strokeWidth={2} aria-hidden="true" />
                  : <Moon size={14} strokeWidth={2} aria-hidden="true" />}
                {theme === "dark" ? "Light mode" : "Dark mode"}
              </button>

              {user ? (
                <>
                  <button
                    type="button"
                    className="user-dropdown-item"
                    role="menuitem"
                    onClick={() => setEditing(true)}
                  >
                    <UserCircle2 size={14} strokeWidth={2} aria-hidden="true" />
                    Profile settings
                  </button>
                  <button
                    type="button"
                    className="user-dropdown-item"
                    role="menuitem"
                    onClick={handleSignOut}
                  >
                    <LogOut size={14} strokeWidth={2} aria-hidden="true" />
                    Sign out
                  </button>
                </>
              ) : (
                configured && (
                  // A link, not a fetch: the flow is a chain of redirects
                  // through Google that ends by setting a cookie.
                  <a className="user-dropdown-item" role="menuitem" href={googleLoginUrl()}>
                    <LogIn size={14} strokeWidth={2} aria-hidden="true" />
                    Sign in with Google
                  </a>
                )
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

/** Name and date of birth - the details asked for after the first
 *  sign-in. Google supplies a name, but it is whatever is on the
 *  account, which is often not what a student wants to be called. */
function ProfileForm({ user, onSaved, onCancel }) {
  const [name, setName] = useState(user.display_name || "");
  const [dob, setDob] = useState(user.date_of_birth || "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  const submit = async (event) => {
    event.preventDefault();
    if (!name.trim()) {
      setError("Please enter a name.");
      return;
    }

    setSaving(true);
    setError(null);
    try {
      onSaved(await saveProfile({ displayName: name.trim(), dateOfBirth: dob }));
    } catch (err) {
      setError(err.message || "Could not save.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <form className="user-profile-form" onSubmit={submit}>
      <label>
        Name
        <input
          type="text"
          value={name}
          onChange={(event) => setName(event.target.value)}
          maxLength={80}
          placeholder="What should we call you?"
          autoFocus
        />
      </label>
      <label>
        Date of birth
        <input
          type="date"
          value={dob}
          onChange={(event) => setDob(event.target.value)}
          // Nobody signing up today was born tomorrow.
          max={new Date().toISOString().slice(0, 10)}
        />
      </label>

      {error && <p className="user-profile-error">{error}</p>}

      <div className="user-profile-actions">
        <button type="button" className="user-profile-cancel" onClick={onCancel}>
          Cancel
        </button>
        <button type="submit" className="user-profile-save" disabled={saving}>
          {saving ? "Saving..." : "Save"}
        </button>
      </div>
    </form>
  );
}
