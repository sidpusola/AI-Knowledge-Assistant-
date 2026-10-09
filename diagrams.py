"""Server-side validation for the structured diagrams the model emits.

The browser validates too (frontend/src/lib/diagramSchema.js) and that
is what actually gates rendering - it has to, because answers stream and
a diagram is only complete once the whole block has arrived. This module
is the server's own check on the finished answer: it says, in the logs,
how often the model produces a diagram the renderer will reject, which
is the number worth watching when the prompt or the model changes.

Keep the rules here in step with the frontend validator. They are
deliberately the same rules in two places: one guards rendering, the
other measures quality.
"""

import json
import logging
import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)

# Mirrors ALLOWED_ICON_NAMES in frontend/src/lib/iconRegistry.js.
ALLOWED_ICONS = {
    "book-open", "box", "brain", "bug", "calculator",
    "calendar", "check", "check-circle-2", "circle-dot", "clock", "cloud",
    "cloud-rain", "code", "cog", "cpu", "database", "download", "droplet",
    "egg", "file-text", "filter", "flag", "flask-conical", "flower-2",
    "git-branch", "globe", "hard-drive", "key", "layers", "leaf",
    "lightbulb", "list-checks", "lock", "mail", "microscope", "monitor",
    "network", "package", "pen-line", "play", "repeat", "refresh-cw",
    "rocket", "search", "send", "server", "settings", "shield",
    "smartphone", "sprout", "sun", "table", "target", "terminal",
    "thermometer", "triangle-alert", "upload", "users", "wind", "worm",
    "x", "x-circle", "zap",
}

# Timelines key their stages by year ("1956"), so a leading digit is
# allowed. Keep in step with ID_RE in the frontend validator.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,39}$")

# Node ids also become Mermaid identifiers when a diagram falls back to
# Mermaid, which is why they are restricted rather than free text.
class DiagramNode(BaseModel):
    id: str
    label: str = Field(min_length=1, max_length=60)
    description: str | None = Field(default=None, max_length=140)
    icon: str | None = None
    role: Literal["start", "process", "decision", "end"] | None = None

    @field_validator("id")
    @classmethod
    def _check_id(cls, value: str) -> str:
        if not _ID_RE.match(value):
            raise ValueError(f"bad node id: {value!r}")
        return value

    @field_validator("icon")
    @classmethod
    def _known_icon(cls, value: str | None) -> str | None:
        # Dropped rather than rejected, exactly as the frontend does - an
        # unknown icon costs a picture, not the diagram.
        return value if value in ALLOWED_ICONS else None


class DiagramEdge(BaseModel):
    source: str
    target: str
    label: str | None = Field(default=None, max_length=40)

    def __init__(self, **data):
        # The model uses from/to about as often as source/target.
        if "source" not in data and "from" in data:
            data["source"] = data.pop("from")
        if "target" not in data and "to" in data:
            data["target"] = data.pop("to")
        super().__init__(**data)


class Diagram(BaseModel):
    diagram_type: Literal["cycle", "timeline", "linear"]
    title: str | None = Field(default=None, max_length=80)
    theme: Literal["minimal", "biology", "technical", "education"] = "minimal"
    nodes: list[DiagramNode] = Field(min_length=1, max_length=24)
    edges: list[DiagramEdge] = Field(default_factory=list, max_length=48)

    @field_validator("nodes")
    @classmethod
    def _unique_ids(cls, nodes: list[DiagramNode]) -> list[DiagramNode]:
        seen = set()
        for node in nodes:
            if node.id in seen:
                raise ValueError(f"duplicate node id: {node.id}")
            seen.add(node.id)
        return nodes

    def check_edges(self) -> list[str]:
        """Edges pointing at nodes that don't exist, plus the filler-node
        check the prompt keeps having to repeat."""
        problems = []
        ids = {node.id for node in self.nodes}
        for edge in self.edges:
            if edge.source not in ids or edge.target not in ids:
                problems.append(f"edge {edge.source} -> {edge.target} names an unknown node")
        for node in self.nodes:
            if node.label.strip().lower() in {"start", "end", "begin", "finish"}:
                problems.append(f'filler node: "{node.label}"')
        return problems


_BLOCK_RE = re.compile(r"```flow-json[^\n]*\n(.*?)```", re.S)


def validate_answer_diagrams(answer: str) -> list[str]:
    """Return a list of problems with the diagrams in one answer.

    Empty list means every diagram in it is fine (or there were none).
    Never raises: a bad diagram is a quality signal, not a request
    failure, and the student still gets the words around it.
    """
    problems: list[str] = []

    for index, block in enumerate(_BLOCK_RE.findall(answer)):
        where = f"diagram {index + 1}"
        try:
            payload = json.loads(block)
        except json.JSONDecodeError as exc:
            problems.append(f"{where}: not valid JSON ({exc.msg})")
            continue

        try:
            diagram = Diagram.model_validate(payload)
        except ValidationError as exc:
            first = exc.errors()[0]
            location = ".".join(str(part) for part in first["loc"])
            problems.append(f"{where}: {location}: {first['msg']}")
            continue

        problems.extend(f"{where}: {problem}" for problem in diagram.check_edges())

    return problems


def log_answer_diagrams(answer: str) -> None:
    """Log any problems. Called on the non-streaming path; see the module
    docstring for why streaming can't go through here."""
    for problem in validate_answer_diagrams(answer):
        logger.warning("[diagram] %s", problem)
