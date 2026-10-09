/**
 * The only icons a diagram may use.
 *
 * Icon names arrive inside model output, so they are untrusted input.
 * This is a closed map rather than a dynamic `lucide[name]` lookup: an
 * unknown or hostile name can only ever miss and fall back, never reach
 * into the icon library. Named imports also let the bundler drop the
 * hundreds of icons that aren't listed here.
 *
 * Keys are lucide's own kebab-case names, which is what the prompt shows
 * the model. Keep ALLOWED_ICON_NAMES (below) in step with the prompt in
 * chains/tutor_chain.py - the model can only pick from what it's told.
 */
import {
  BookOpen, Box, Brain, Bug, Calculator, Calendar, Check,
  CheckCircle2, CircleDot, Clock, Cloud, CloudRain, Code, Cog, Cpu,
  Database, Download, Droplet, Egg, FileText, Filter, Flag, FlaskConical,
  Flower2, GitBranch, Globe, HardDrive, Key, Layers, Leaf, Lightbulb,
  ListChecks, Lock, Mail, Microscope, Monitor, Network, Package, PenLine,
  Play, Repeat, RefreshCw, Rocket, Search, Send, Server, Settings, Shield,
  Smartphone, Sprout, Sun, Table, Target, Terminal, Thermometer,
  TriangleAlert, Upload, Users, Wind, Worm, X, XCircle, Zap,
} from "lucide-react";

const ICONS = {
  // No "arrow-right": it is a connector glyph, and when the model
  // found no fitting icon for a topic it gave every single node that
  // one, which looks like a bug. Better no icon than six identical
  // arrows.
  "book-open": BookOpen, box: Box, brain: Brain,
  bug: Bug, calculator: Calculator, calendar: Calendar, check: Check,
  "check-circle-2": CheckCircle2, "circle-dot": CircleDot, clock: Clock,
  cloud: Cloud, "cloud-rain": CloudRain, code: Code, cog: Cog, cpu: Cpu,
  database: Database, download: Download, droplet: Droplet, egg: Egg,
  "file-text": FileText, filter: Filter, flag: Flag,
  "flask-conical": FlaskConical, "flower-2": Flower2, "git-branch": GitBranch,
  globe: Globe, "hard-drive": HardDrive, key: Key, layers: Layers, leaf: Leaf,
  lightbulb: Lightbulb, "list-checks": ListChecks, lock: Lock, mail: Mail,
  microscope: Microscope, monitor: Monitor, network: Network, package: Package,
  "pen-line": PenLine, play: Play, repeat: Repeat, "refresh-cw": RefreshCw,
  rocket: Rocket, search: Search, send: Send, server: Server,
  settings: Settings, shield: Shield, smartphone: Smartphone, sprout: Sprout,
  sun: Sun, table: Table, target: Target, terminal: Terminal,
  thermometer: Thermometer, "triangle-alert": TriangleAlert, upload: Upload,
  users: Users, wind: Wind, worm: Worm, x: X, "x-circle": XCircle, zap: Zap,
};

/** Shown to the model in the prompt; exported so the two can't drift. */
export const ALLOWED_ICON_NAMES = Object.keys(ICONS);

/** Own-property lookup only, so "constructor" or "__proto__" can't match. */
export function getIcon(name) {
  if (typeof name !== "string") return null;
  return Object.prototype.hasOwnProperty.call(ICONS, name) ? ICONS[name] : null;
}
