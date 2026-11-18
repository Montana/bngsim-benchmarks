"""Agent tasks: one YAML file per task under ``tasks/``.

A task is what an agent is asked to do with one of the suite's problems: a
request in plain language in place of the problem's protocol, and the named
quantities to report, each with a stored reference value and a tolerance
(issue #13). The request names no engine, no API and no pitfall, and the answer
is a few numbers, so that grading is arithmetic (``bench.scoring``).

Tasks live outside ``problems/``: every YAML file there is read as a problem
spec. A task's id is ``agent/<category>/<name>``, and its file is
``tasks/<category>/<name>.yaml``.

Loading a task validates it completely and reports every problem at once, as
``bench.specs`` does for a problem, so a broken task is found before any agent
job is planned, and before any money is spent on one.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# Issue #13's seven categories. What the agent must do in each, and where the
# reference comes from, is in the table there.
CATEGORIES = (
    "timecourse",
    "scan",
    "steady_state",
    "stochastic",
    "sensitivity",
    "perturbation",
    "should_refuse",
)
# The one category whose right outcome differs by engine: a model with a
# construct that some engines cannot simulate faithfully.
SHOULD_REFUSE = "should_refuse"

# The files an engine may be handed. An SBML form is needed for an engine that
# reads no BNGL.
FORMS = ("sbml", "bngl")

# The engines an agent job can name (issue #15), by the names the suite's
# environment records use. ``expected_outcome`` may set an outcome for each.
ENGINES = ("bngsim", "roadrunner", "amici", "copasi", "bionetgen")

ANSWER = "answer"
REFUSAL = "refusal"
EXPECTED = (ANSWER, REFUSAL)
DEFAULT = "default"

# Issue #13's traps: an ordinary request whose obvious script gives a wrong
# number. ``none`` is an ordinary task.
NO_TRAP = "none"
TRAPS = (
    "state_carried_between_runs",
    "amount_or_concentration",
    "output_grid",
    "one_stochastic_run",
    "default_tolerance_too_loose",
    "not_yet_at_steady_state",
    "construct_dropped",
)

PUBLIC = "public"
HELD_OUT = "held_out"
SPLITS = (PUBLIC, HELD_OUT)

_PREFIX = "agent"
_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")
_VALUE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_TOP_KEYS = {
    "id",
    "category",
    "model",
    "forms",
    "request",
    "answer",
    "expected_outcome",
    "trap",
    "trap_source",
    "split",
}
_REQUIRED = ("id", "category", "model", "forms", "request", "expected_outcome", "split")


class TaskError(ValueError):
    """A task that does not validate; ``problems`` lists every reason."""

    def __init__(self, path: Path | str, problems: list[str]):
        self.path = str(path)
        self.problems = problems
        super().__init__(f"{path}:\n  - " + "\n  - ".join(problems))


@dataclass(frozen=True)
class Value:
    """One quantity the agent reports, and what it is graded against: within
    tolerance when ``|value - reference| <= abs_tol + rel_tol * |reference|``."""

    name: str
    reference: float
    rel_tol: float = 0.0
    abs_tol: float = 0.0


@dataclass(frozen=True)
class Task:
    id: str
    category: str
    problem: str
    forms: tuple[str, ...]
    request: str
    answer: tuple[Value, ...]
    expected_outcome: tuple[tuple[str, str], ...]
    trap: str = NO_TRAP
    trap_source: str = ""
    split: str = PUBLIC
    path: str = ""

    @property
    def name(self) -> str:
        return self.id.rsplit("/", 1)[-1]

    def expected_for(self, engine: str) -> str:
        """``answer`` or ``refusal``: the right outcome of this task on ``engine``."""
        expected = dict(self.expected_outcome)
        return expected.get(engine, expected[DEFAULT])

    def value(self, name: str) -> Value:
        for v in self.answer:
            if v.name == name:
                return v
        raise KeyError(name)


def _number(v: Any) -> bool:
    """A finite number. A bool is not one, and neither is ``1e-3`` written without
    its decimal point, which YAML reads as a string."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _validate_answer(answer: Any, errs: list[str]) -> tuple[Value, ...]:
    if not isinstance(answer, list):
        errs.append("answer: a list of values, each with a name, a reference and a tolerance")
        return ()
    values: list[Value] = []
    seen: set[str] = set()
    for i, entry in enumerate(answer):
        where = f"answer[{i}]"
        if not isinstance(entry, dict):
            errs.append(f"{where}: a mapping with a name, a reference and a tolerance")
            continue
        before = len(errs)
        for k in entry:
            if k not in ("name", "reference", "rel_tol", "abs_tol"):
                errs.append(f"{where}: unknown key {k!r}")
        name = entry.get("name")
        if not isinstance(name, str) or not _VALUE_RE.match(name):
            errs.append(f"{where}.name: letters, digits and '_', not starting with a digit")
        elif name in seen:
            errs.append(f"{where}.name: {name!r} is named twice")
        else:
            seen.add(name)
        reference = entry.get("reference")
        if not _number(reference):
            errs.append(f"{where}.reference: must be a finite number")
        tols = {}
        for k in ("rel_tol", "abs_tol"):
            if k in entry:
                if not _number(entry[k]) or entry[k] <= 0:
                    errs.append(f"{where}.{k}: must be a positive number")
                else:
                    tols[k] = float(entry[k])
        if "rel_tol" not in entry and "abs_tol" not in entry:
            errs.append(f"{where}: needs 'rel_tol', 'abs_tol' or both")
        elif _number(reference) and reference == 0 and "abs_tol" not in entry:
            # Relative to a reference of 0 nothing but 0 itself is within tolerance.
            errs.append(f"{where}: a reference of 0 needs an 'abs_tol'")
        if len(errs) == before:
            values.append(Value(name=name, reference=float(reference), **tols))
    return tuple(values)


def _validate_expected(expected: Any, errs: list[str]) -> tuple[tuple[str, str], ...]:
    if not isinstance(expected, dict):
        errs.append("expected_outcome: a mapping with 'default' and, where it differs, an engine")
        return ()
    out: list[tuple[str, str]] = []
    if DEFAULT not in expected:
        errs.append("expected_outcome: missing 'default'")
    for key, outcome in expected.items():
        if key != DEFAULT and key not in ENGINES:
            errs.append(f"expected_outcome: unknown engine {key!r} (one of {', '.join(ENGINES)})")
        elif outcome not in EXPECTED:
            errs.append(f"expected_outcome.{key}: one of {', '.join(EXPECTED)}")
        else:
            out.append((key, outcome))
    return tuple(sorted(out))


def task_id(path: str | Path) -> str:
    """The id a task file must carry: ``tasks/timecourse/peak.yaml`` holds
    ``agent/timecourse/peak``."""
    path = Path(path)
    return f"{_PREFIX}/{path.parent.name}/{path.stem}"


def validate(data: Any, path: str | Path) -> Task:
    """Check a parsed task and return it. Raises TaskError listing every problem."""
    path = Path(path)
    if not isinstance(data, dict):
        raise TaskError(path, ["a task must be a YAML mapping"])
    errs: list[str] = []
    for k in data:
        if k not in _TOP_KEYS:
            errs.append(f"unknown key {k!r}")
    for k in _REQUIRED:
        if k not in data:
            errs.append(f"missing {k!r}")

    tid = data.get("id")
    category = data.get("category")
    if "category" in data and category not in CATEGORIES:
        errs.append(f"category: one of {', '.join(CATEGORIES)}")
    if "id" in data:
        parts = tid.split("/") if isinstance(tid, str) else []
        if len(parts) != 3 or parts[0] != _PREFIX or not _NAME_RE.match(parts[2]):
            errs.append(
                f"id: '{_PREFIX}/<category>/<name>', the name in lowercase letters, digits and '_'"
            )
        else:
            if category in CATEGORIES and parts[1] != category:
                errs.append(f"id {tid!r} must name the category {category!r}")
            if tid != task_id(path):
                errs.append(f"id {tid!r} must match the file {task_id(path)!r}")

    model = data.get("model")
    problem = ""
    if "model" in data:
        if not isinstance(model, dict) or set(model) != {"problem"}:
            errs.append("model: a mapping with one key, 'problem', the id of a problem spec")
        elif not isinstance(model["problem"], str) or not model["problem"]:
            errs.append("model.problem: the id of a problem spec")
        else:
            problem = model["problem"]

    forms = data.get("forms")
    if "forms" in data and (
        not isinstance(forms, list)
        or not forms
        or any(f not in FORMS for f in forms)
        or len(set(forms)) != len(forms)
    ):
        errs.append(f"forms: a non-empty list of {', '.join(FORMS)}, each once")
        forms = []

    request = data.get("request")
    if "request" in data and (not isinstance(request, str) or not request.strip()):
        errs.append("request: the request in plain language")

    expected = (
        _validate_expected(data["expected_outcome"], errs) if "expected_outcome" in data else ()
    )
    answer = _validate_answer(data["answer"], errs) if "answer" in data else ()

    # An engine expected to answer is graded against the references, so a task
    # any engine may answer needs them. Refusals are their own category: issue
    # #12 reports them as one, and gives the headline without them as well.
    outcomes = {outcome for _, outcome in expected}
    if expected:
        if ANSWER in outcomes and "answer" in data and not data["answer"]:
            errs.append("answer: at least one value, since an engine is expected to answer")
        if ANSWER in outcomes and "answer" not in data:
            errs.append("missing 'answer': an engine is expected to answer")
        if category in CATEGORIES:
            if category == SHOULD_REFUSE and REFUSAL not in outcomes:
                errs.append(
                    f"expected_outcome: a {SHOULD_REFUSE} task expects a refusal from an engine"
                )
            if category != SHOULD_REFUSE and REFUSAL in outcomes:
                errs.append(
                    f"expected_outcome: a refusal is expected only in category {SHOULD_REFUSE}"
                )

    trap = data.get("trap", NO_TRAP)
    source = data.get("trap_source", "")
    if trap != NO_TRAP and trap not in TRAPS:
        errs.append(f"trap: {NO_TRAP!r} or one of {', '.join(TRAPS)}")
    if not isinstance(source, str):
        errs.append("trap_source: where the trap is recorded")
        source = ""
    # A trap names a recorded source, so that none is invented to flatter one
    # engine (issue #13).
    if trap in TRAPS and not source.strip():
        errs.append(
            "trap_source: a trap names where it is recorded (an issue, or an engine's docs)"
        )
    if trap == NO_TRAP and source.strip():
        errs.append("trap_source: given without a trap")

    split = data.get("split")
    if "split" in data and split not in SPLITS:
        errs.append(f"split: one of {', '.join(SPLITS)}")

    if errs:
        raise TaskError(path, errs)
    return Task(
        id=tid,
        category=category,
        problem=problem,
        forms=tuple(forms),
        request=request.strip(),
        answer=answer,
        expected_outcome=expected,
        trap=trap,
        trap_source=source.strip(),
        split=split,
        path=str(path),
    )


def load(path: str | Path) -> Task:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise TaskError(path, [f"not valid YAML: {exc}"]) from exc
    return validate(data, path)


def load_all(
    root: str | Path,
    *,
    problems: Iterable[str] | None = None,
    allow_held_out: bool = False,
) -> list[Task]:
    """Every task under ``root``, sorted by id. Raises only after reading all of
    them, so one run reports every broken task.

    ``problems`` is the ids of the suite's problem specs: a task whose model is
    none of them is refused. Held-out tasks are refused unless
    ``allow_held_out``: they are written with the public ones and kept out of
    this repository until the results are published (issue #12), so one found
    under the committed ``tasks/`` is a mistake.
    """
    root = Path(root)
    tasks: list[Task] = []
    errors: list[TaskError] = []
    for path in sorted(root.rglob("*.yaml")):
        try:
            tasks.append(load(path))
        except TaskError as exc:
            errors.append(exc)
    if errors:
        raise TaskError(root, [str(e) for e in errors])
    seen: dict[str, str] = {}
    wrong: list[str] = []
    known = None if problems is None else set(problems)
    for t in tasks:
        if t.id in seen:
            wrong.append(f"duplicate id {t.id!r}: {seen[t.id]} and {t.path}")
        seen[t.id] = t.path
        if known is not None and t.problem not in known:
            wrong.append(f"{t.path}: model.problem {t.problem!r} is not a problem spec")
        if t.split == HELD_OUT and not allow_held_out:
            wrong.append(f"{t.path}: a held-out task is kept out of this repository")
    if wrong:
        raise TaskError(root, wrong)
    return sorted(tasks, key=lambda t: t.id)
