"""Versioned prompts.

A prompt is code: changing a sentence moves the eval numbers as surely as
changing a function. So each prompt registered here has a name and a
version, and every trace, workflow result and eval report records which
version produced it - "faithfulness fell from 0.92 to 0.81" is only
actionable if you know which prompt text each number came from.

The version is a label a person bumps; the fingerprint is a hash of the
text that cannot lie. prompts.lock.json records both, and --check fails
when a prompt's text changed but its version did not - the one mistake
that would make two different prompts report the same version.

    python -m llm.prompt_registry --check     verify the lock file
    python -m llm.prompt_registry --update    record the current prompts
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

LOCK_FILE = Path(__file__).resolve().parent.parent / "prompts.lock.json"


@dataclass(frozen=True)
class PromptVersion:
    name: str
    version: str
    messages: tuple[tuple[str, str], ...]

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(self.messages, ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"

    def template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(list(self.messages))


_REGISTRY: dict[str, PromptVersion] = {}


def register(name: str, version: str, messages: list[tuple[str, str]]) -> PromptVersion:
    prompt = PromptVersion(name, version, tuple(tuple(m) for m in messages))
    _REGISTRY[name] = prompt
    return prompt


def versions() -> dict[str, str]:
    """name -> "name@version", for traces and reports."""
    return {name: prompt.id for name, prompt in sorted(_REGISTRY.items())}


def _load_all() -> None:
    # Registration happens at import; these are the modules that register.
    import graph.evaluation  # noqa: F401
    import graph.rag_workflow  # noqa: F401


def check() -> list[str]:
    """Problems with the lock file, empty when it matches the code."""
    _load_all()
    locked = json.loads(LOCK_FILE.read_text(encoding="utf-8")) if LOCK_FILE.exists() else {}
    problems = []
    for name, prompt in sorted(_REGISTRY.items()):
        entry = locked.get(name)
        if entry is None:
            problems.append(f"{name}: not in the lock file - run --update")
        elif entry["version"] == prompt.version and entry["fingerprint"] != prompt.fingerprint:
            problems.append(f"{name}@{prompt.version}: text changed but version did not - bump the version")
        elif entry["version"] != prompt.version:
            problems.append(f"{name}: version {entry['version']} -> {prompt.version} not recorded - run --update")
    return problems


def update() -> None:
    _load_all()
    LOCK_FILE.write_text(json.dumps(
        {name: {"version": p.version, "fingerprint": p.fingerprint} for name, p in sorted(_REGISTRY.items())},
        indent=2,
    ) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.path.insert(0, str(LOCK_FILE.parent))
    # Run as a script this file is "__main__", a different module from the
    # llm.prompt_registry the prompts register into. Use that one.
    from llm import prompt_registry as registry

    if "--update" in sys.argv:
        registry.update()
        print(f"recorded {len(registry._REGISTRY)} prompts in {LOCK_FILE.name}")
    else:
        issues = registry.check()
        for issue in issues:
            print("  FAIL", issue)
        print("  prompts match the lock file" if not issues else f"  {len(issues)} problem(s)")
        raise SystemExit(1 if issues else 0)
