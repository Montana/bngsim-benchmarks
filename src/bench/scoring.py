"""Scoring agent jobs: one outcome per job, and the rates the study reports.

An agent job (issue #15) ends in ``answer.json``. ``grade`` turns that file and
the task's stored references into one outcome from a fixed vocabulary (issue
#14), by arithmetic alone. The vocabulary separates the outcome that does harm,
a wrong number reported as a result, from the ones that do not.

The headline is the **silently-wrong rate**: ``silently_wrong`` plus
``unfaithful_answer``, over all jobs. It is always reported beside the
**completion rate** (``correct`` plus ``correct_refusal``): a tool can reach
zero silent errors by never answering, and the pair is what shows that it did
not.

The unit of analysis is the task. Repetitions are nested within a task and are
not independent samples, so a rate is the mean over tasks of each task's own
rate, and its interval comes from a bootstrap that resamples whole tasks.
Engines are compared on the same tasks, as paired differences.

"Within tolerance" is the vendored parity core's mixed tolerance,
``|x - y| <= atol + rtol * |y|`` with the reference as ``y``, so that it means
here what it means in the correctness jobs.
"""

from __future__ import annotations

import json
import math
import operator
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from . import tasks as _tasks

# --------------------------------------------------------------------------- #
# Outcomes
# --------------------------------------------------------------------------- #


class Outcome(str, Enum):
    # The agent answered, and every value is within its tolerance.
    CORRECT = "correct"
    # The task's expected outcome for this engine is a refusal, and the agent
    # refused.
    CORRECT_REFUSAL = "correct_refusal"
    # The agent answered, and a value is outside its tolerance.
    SILENTLY_WRONG = "silently_wrong"
    # The expected outcome is a refusal, and the agent answered anyway: it
    # dropped the construct, or switched on an approximation.
    UNFAITHFUL_ANSWER = "unfaithful_answer"
    # The agent refused a task the engine can do.
    WRONG_REFUSAL = "wrong_refusal"
    # No answer: it gave up, ran out of budget, or stopped at an error.
    FAILURE = "failure"
    # No answer file, a file that does not parse or does not keep the contract,
    # or an engine other than the one named.
    INVALID = "invalid"

    def __str__(self) -> str:
        return self.value


# Column order for a report.
ALL = tuple(Outcome)
# The headline: a wrong result reported as a result.
SILENT = frozenset({Outcome.SILENTLY_WRONG, Outcome.UNFAITHFUL_ANSWER})
# Reported beside it, always.
COMPLETE = frozenset({Outcome.CORRECT, Outcome.CORRECT_REFUSAL})
# Each of these gets a cause, read from the transcript.
NEEDS_CAUSE = frozenset(
    {Outcome.SILENTLY_WRONG, Outcome.UNFAITHFUL_ANSWER, Outcome.WRONG_REFUSAL, Outcome.FAILURE}
)


def tally(outcomes: Iterable[Outcome | str]) -> dict[str, int]:
    """How many jobs ended in each outcome, in report order, zeros included."""
    counts = Counter(Outcome(o) for o in outcomes)
    return {o.value: counts.get(o, 0) for o in ALL}


class Cause(str, Enum):
    """Why a job went wrong, coded from its transcript (issue #14). Each says
    where a fix would go: the engine's docs, its errors, its API, or the engine."""

    WRONG_CALL = "wrong_call"
    MISUSE_START_STATE = "misuse_start_state"
    MISUSE_QUANTITY = "misuse_quantity"
    MISUSE_GRID_OR_TIME = "misuse_grid_or_time"
    MISUSE_STOCHASTIC = "misuse_stochastic"
    MISUSE_ACCURACY = "misuse_accuracy"
    ENVIRONMENT = "environment"
    ENGINE_DEFECT = "engine_defect"
    AFTER_THE_RUN = "after_the_run"
    TASK_MISREAD = "task_misread"

    def __str__(self) -> str:
        return self.value


# --------------------------------------------------------------------------- #
# One value against its reference
# --------------------------------------------------------------------------- #


def _as_fingerprint(x: float) -> dict[str, Any]:
    """One number in the shape ``fingerprint_distance`` compares."""
    return {
        "n_time": 1,
        "t_last": 1.0,
        "vars": {"value": {stat: x for stat in ("last", "min", "max", "mean")}},
    }


def miss(value: float, reference: float, *, rel_tol: float = 0.0, abs_tol: float = 0.0) -> float:
    """How far ``value`` is from ``reference``, in units of the tolerance:
    ``|value - reference| / (abs_tol + rel_tol * |reference|)``. At most 1 is
    within it; inf for a value that is not finite.

    The size is kept, not only the verdict: a value just outside its tolerance
    and one wrong by a factor of 20 are both ``silently_wrong``, and a report
    can say which.
    """
    if not math.isfinite(reference):
        raise ValueError(f"reference {reference!r} is not finite")
    if rel_tol < 0 or abs_tol < 0 or abs_tol + rel_tol * abs(reference) <= 0:
        raise ValueError("a value needs a tolerance that is positive at its reference")
    value = float(value)
    if not math.isfinite(value):
        return math.inf
    # Imported here, as capabilities/ode.py does: the parity core needs bngsim,
    # and nothing else in this module does.
    from ._vendor.parity_core.fingerprint import fingerprint_distance

    return fingerprint_distance(
        _as_fingerprint(value), _as_fingerprint(float(reference)), rel_tol, abs_tol
    )


def within(value: float, reference: float, *, rel_tol: float = 0.0, abs_tol: float = 0.0) -> bool:
    return miss(value, reference, rel_tol=rel_tol, abs_tol=abs_tol) <= 1.0


# --------------------------------------------------------------------------- #
# The answer file
# --------------------------------------------------------------------------- #

_ANSWER_KEYS = {"outcome", "values", "checked", "check", "notes"}
_ANSWER_OUTCOMES = (_tasks.ANSWER, _tasks.REFUSAL, "failure")
_SHOWN = 60  # how much of an agent's own text a problem quotes
_MAX_SLIPS = 20  # how many slips a grade lists before it counts the rest
_MAX_DIGITS = 400  # an integer longer than this is read as infinite, unconverted


def _shown(text: Any) -> str:
    """An agent's text, cut short: a problem names the entry, and does not carry
    whatever the agent put there into the result."""
    text = str(text)
    quoted = repr(text[:_SHOWN])
    cut = len(text) > _SHOWN or len(quoted) > _SHOWN + 20
    return quoted[: _SHOWN + 20] + ("..." if cut else "")


def _as_float(v: Any) -> float | None:
    """A JSON number as a float, or None for what is not one. An integer too
    large for a float is infinite, which no tolerance admits."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        return float(v)
    except OverflowError:
        return math.inf if v > 0 else -math.inf


def _number_written(v: Any) -> tuple[float | None, str]:
    """``(number, slip)`` for one entry of ``values``. A number is itself. A
    number written as text, ``"0.5"``, or alone in a list, ``[0.5]``, is read as
    the number it holds, with a slip: quoted, a wrong number would otherwise
    read as a value left out. Anything else is not a number."""
    number = _as_float(v)
    if number is not None:
        return number, ""
    if isinstance(v, str):
        try:
            return float(v.strip()), "a number written as text"
        except ValueError:
            return None, "not a number"
    if isinstance(v, list) and len(v) == 1 and _as_float(v[0]) is not None:
        return _as_float(v[0]), "a number written as a list"
    return None, "not a number"


class _Object(dict):
    """A JSON object that remembers the entries a repeated key overwrote."""

    repeated: tuple[tuple[str, Any], ...] = ()


def _object(pairs: list[tuple[str, Any]]) -> _Object:
    obj = _Object()
    repeated = []
    for key, value in pairs:
        if key in obj:
            repeated.append((key, obj[key]))
        obj[key] = value
    obj.repeated = tuple(repeated)
    return obj


def _parse_int(digits: str) -> int | float:
    # Python refuses to convert an integer of more than 4300 digits, and the
    # file is valid JSON all the same. No tolerance admits a number that long.
    if len(digits) > _MAX_DIGITS:
        return -math.inf if digits.startswith("-") else math.inf
    return int(digits)


@dataclass(frozen=True)
class Answer:
    """``answer.json``, the contract between the agent and the grader (issue
    #15). ``values`` holds every number the file gives for a name, more than one
    where the file names it twice. ``slips`` are what is the matter with a file
    that could still be read."""

    outcome: str
    values: tuple[tuple[str, float], ...] = ()
    checked: bool = False
    check: str = ""
    notes: str = ""
    slips: tuple[str, ...] = ()


def parse_answer(doc: Any) -> tuple[Answer | None, list[str]]:
    """``(answer, problems)`` from the bytes, the text or the parsed content of
    ``answer.json``.

    ``answer`` is None only where the file cannot be read as an answer at all:
    there is none, it does not parse, it is not an object, it names no outcome,
    or it says ``answer`` and its values are not an object. Those are
    ``invalid``. Anything less is read, and kept in ``Answer.slips``: an unknown
    key, a ``checked`` that is not true or false, a value that is not a number,
    a key given twice. A slip must not move a job out of the headline: a file
    with a wrong number in it and a stray key reported a wrong number.

    A key given twice is read both ways. JSON parsers keep the last, so a file
    could show a wrong number first and a right one after; every number given
    for a name is graded, and an ``outcome`` of ``answer`` given anywhere
    stands.
    """
    if doc is None:
        return None, ["no answer file"]
    if isinstance(doc, bytes):
        try:
            doc = doc.decode(json.detect_encoding(doc))
        except UnicodeDecodeError as exc:
            return None, [f"not valid JSON: {exc}"]
    if isinstance(doc, str):
        try:
            doc = json.loads(
                doc.removeprefix("\ufeff"), object_pairs_hook=_object, parse_int=_parse_int
            )
        except (ValueError, RecursionError) as exc:
            reason = "nested too deeply" if isinstance(exc, RecursionError) else str(exc)
            return None, [f"not valid JSON: {reason}"]
    if not isinstance(doc, dict):
        return None, ["the answer must be a JSON object"]
    repeated = list(getattr(doc, "repeated", ()))
    slips = [
        f"{_shown(k)[1:-1]}: given more than once" for k in dict.fromkeys(k for k, _ in repeated)
    ]

    outcomes = [doc.get("outcome")] + [v for k, v in repeated if k == "outcome"]
    if not all(isinstance(o, str) and o in _ANSWER_OUTCOMES for o in outcomes):
        return None, [f"outcome: one of {', '.join(_ANSWER_OUTCOMES)}"]
    # Given twice and not alike, the one that reports numbers stands.
    outcome = _tasks.ANSWER if _tasks.ANSWER in outcomes else outcomes[0]

    slips += [f"unknown key {_shown(k)}" for k in doc if k not in _ANSWER_KEYS]
    pairs: list[tuple[str, float]] = []
    for values in [doc.get("values", {})] + [v for k, v in repeated if k == "values"]:
        if not isinstance(values, dict):
            if outcome == _tasks.ANSWER:
                return None, ["values: an object of name and number"]
            # A refusal or a failure reports no values, and null says so.
            if values is not None:
                slips.append("values: not an object")
            continue
        for name, v in list(values.items()) + list(getattr(values, "repeated", ())):
            number, slip = _number_written(v)
            if slip:
                slips.append(f"values.{_shown(name)[1:-1]}: {slip}")
            if number is not None:
                pairs.append((str(name), number))
        slips += [
            f"values.{_shown(k)[1:-1]}: given more than once"
            for k in dict.fromkeys(k for k, _ in getattr(values, "repeated", ()))
        ]
    checked = doc.get("checked", False)
    if not isinstance(checked, bool):
        slips.append("checked: not true or false")
        checked = False
    text = {}
    for k in ("check", "notes"):
        text[k] = doc.get(k)
        if text[k] is None:
            text[k] = ""
        elif not isinstance(text[k], str):
            slips.append(f"{k}: not text")
            text[k] = ""
    if len(slips) > _MAX_SLIPS:
        slips = slips[:_MAX_SLIPS] + [f"and {len(slips) - _MAX_SLIPS} more"]
    return (
        Answer(
            outcome=outcome,
            values=tuple(pairs),
            checked=checked,
            check=text["check"],
            notes=text["notes"],
            slips=tuple(slips),
        ),
        [],
    )


# --------------------------------------------------------------------------- #
# Grading one job
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Grade:
    outcome: Outcome
    # For each value the task asks for and the agent reported, how far it is
    # from its reference in units of its tolerance (``miss``); the furthest,
    # where the file gave a name more than one number.
    misses: tuple[tuple[str, float], ...] = ()
    # Why the job is ``invalid``, the values it left out, and the slips in a
    # file that was read all the same; empty for a file that keeps the contract.
    problems: tuple[str, ...] = ()
    # What the agent said of its own answer (``checked`` in the answer file).
    # The transcript confirms it; this is only the claim.
    checked: bool = False

    @property
    def worst_miss(self) -> float | None:
        return max((m for _, m in self.misses), default=None)


def _audit(engines_used: Collection[str] | None) -> set[str] | None:
    if engines_used is None:
        return None
    if isinstance(engines_used, (str, bytes)):
        raise TypeError("engines_used is a collection of engine names, not one name")
    used = set(engines_used)
    unknown = sorted(str(name) for name in used if name not in _tasks.ENGINES)
    if unknown:
        # The harness maps what a job imported to these names (Tellurium is
        # roadrunner); a name that is none of them is its mistake, not a job's.
        raise ValueError(
            f"engines_used: {', '.join(map(_shown, unknown))} is not one of "
            f"{', '.join(_tasks.ENGINES)}"
        )
    return used


def grade(
    task: _tasks.Task,
    engine: str,
    answer: Any,
    *,
    engines_used: Collection[str] | None = None,
) -> Grade:
    """The outcome of one job: ``task`` on ``engine``, ending in ``answer``.

    ``answer`` is the bytes, the text or the parsed content of ``answer.json``,
    or None where the agent wrote none. ``engine`` is the engine the job named;
    in the ``free`` arm, where none is named, it is the one the agent chose.

    ``engines_used`` is the import audit: the engines the job imported, by the
    names in ``tasks.ENGINES``. A job that used an engine other than the one
    named is ``invalid``, whatever it answered, and so is one that answered
    without importing the one named (issue #15). One that imported none and did
    not answer is graded on what it wrote: an agent that stopped at a missing
    compiler failed, and one that read the model and refused, refused.

    Where the agent answered and the engine was expected to, any number the
    file gives for a value the task asks for, outside its tolerance, makes the
    job ``silently_wrong``: whatever else is the matter with the file, another
    value left out, a stray key, the number in quotes, the name given twice
    with a right number after. An answer that leaves a value out and has every
    number it does give within tolerance does not keep the contract, and is
    ``invalid``. So is one whose number cannot be read as one at all, an object
    or a word, or sits under another name than the task's.

    A refusal or a failure is graded as that; numbers beside it are not read.
    """
    if engine not in _tasks.ENGINES:
        raise ValueError(f"unknown engine {engine!r} (one of {', '.join(_tasks.ENGINES)})")
    used = _audit(engines_used)
    parsed, problems = parse_answer(answer)
    if parsed is None:
        return Grade(Outcome.INVALID, problems=tuple(problems))

    expected = task.expected_for(engine)
    answered = parsed.outcome == _tasks.ANSWER
    asked = {v.name for v in task.answer}
    misses: tuple[tuple[str, float], ...] = ()
    slips = parsed.slips
    if answered and expected == _tasks.ANSWER:
        worst: dict[str, float] = {}
        for name, number in parsed.values:
            if name in asked:
                v = task.value(name)
                m = miss(number, v.reference, rel_tol=v.rel_tol, abs_tol=v.abs_tol)
                worst[name] = max(worst.get(name, 0.0), m)
        misses = tuple((v.name, worst[v.name]) for v in task.answer if v.name in worst)
        missing = tuple(f"values: missing {v.name!r}" for v in task.answer if v.name not in worst)
        others = sorted({name for name, _ in parsed.values if name not in asked})
        extra = tuple(f"values.{_shown(n)[1:-1]}: not a value the task asks for" for n in others)
        slips = (missing + slips + extra[:_MAX_SLIPS])[: 2 * _MAX_SLIPS + len(missing)]
    kept = {"misses": misses, "checked": parsed.checked}

    if used is not None and used - {engine}:
        why = f"the job was to use {engine}, and also used: {', '.join(sorted(used - {engine}))}"
        return Grade(Outcome.INVALID, problems=(why,) + slips, **kept)
    if answered and used is not None and engine not in used:
        why = f"the job answered without importing {engine}"
        return Grade(Outcome.INVALID, problems=(why,) + slips, **kept)
    if parsed.outcome == "failure":
        return Grade(Outcome.FAILURE, problems=slips, **kept)
    if parsed.outcome == _tasks.REFUSAL:
        outcome = Outcome.CORRECT_REFUSAL if expected == _tasks.REFUSAL else Outcome.WRONG_REFUSAL
        return Grade(outcome, problems=slips, **kept)
    if expected == _tasks.REFUSAL:
        return Grade(Outcome.UNFAITHFUL_ANSWER, problems=slips, **kept)
    if any(m > 1.0 for _, m in misses):
        return Grade(Outcome.SILENTLY_WRONG, problems=slips, **kept)
    if len(misses) < len(task.answer):
        return Grade(Outcome.INVALID, problems=slips, **kept)
    return Grade(Outcome.CORRECT, problems=slips, **kept)


# --------------------------------------------------------------------------- #
# Rates, with the task as the unit
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Scored:
    """One graded job, as the rates need it."""

    task: str
    category: str
    engine: str
    arm: str
    agent: str
    repetition: int
    outcome: Outcome
    checked: bool = False
    cause: Cause | None = None
    # The largest miss among its values, in units of the tolerance (``miss``):
    # what a second headline at a looser tolerance would be taken from.
    worst_miss: float | None = None


@dataclass(frozen=True)
class Rate:
    """A rate over tasks: the mean of each task's own rate, with a bootstrap
    interval from resampling whole tasks. ``low`` and ``high`` are None with
    fewer than two tasks, where there is nothing to resample."""

    value: float
    low: float | None
    high: float | None
    tasks: int
    jobs: int


def task_rates(jobs: Iterable[Scored], outcomes: Collection[Outcome]) -> dict[str, float]:
    """For each task, the fraction of its jobs whose outcome is in ``outcomes``.

    Repetitions of one task are not independent samples: three wrong answers to
    one task are one task done badly, and this is where they become one number.
    """
    outcomes = _outcomes(outcomes)
    hits: Counter[str] = Counter()
    total: Counter[str] = Counter()
    for job in jobs:
        total[job.task] += 1
        hits[job.task] += Outcome(job.outcome) in outcomes
    return {task: hits[task] / total[task] for task in sorted(total)}


def _whole(value: Any, name: str, least: int) -> int:
    """``value`` as an integer of at least ``least`` (a numpy integer is one; a
    bool, a float and None are not)."""
    what = {"seed": "seed: a whole number from 0, so that the same jobs give the same interval"}
    message = what.get(name, f"{name}: a positive integer")
    if isinstance(value, bool):
        raise ValueError(message)
    try:
        value = operator.index(value)
    except TypeError:
        raise ValueError(message) from None
    if value < least:
        raise ValueError(message)
    return value


def _outcomes(outcomes: Collection[Outcome | str]) -> frozenset[Outcome]:
    """The outcomes a rate counts, as a set. Taken once: a generator would be
    spent on the first job, and a rate of 0 returned for the rest."""
    if isinstance(outcomes, (str, bytes)):
        raise TypeError("outcomes: a collection of outcomes, not one name")
    return frozenset(Outcome(o) for o in outcomes)


def _interval(
    values: np.ndarray, n_boot: int, confidence: float, seed: int
) -> tuple[float | None, float | None]:
    """The percentile interval of the mean of ``values``, one per task, from
    ``n_boot`` resamples of whole tasks. None with fewer than two tasks."""
    seed, n_boot = _whole(seed, "seed", 0), _whole(n_boot, "n_boot", 1)
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence: between 0 and 1")
    if not 0 < confidence < 1:
        raise ValueError("confidence: between 0 and 1")
    if values.size < 2:
        return None, None
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    lo, hi = np.quantile(values[idx].mean(axis=1), [(1 - confidence) / 2, 1 - (1 - confidence) / 2])
    return float(lo), float(hi)


def rate(
    jobs: Iterable[Scored],
    outcomes: Collection[Outcome],
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Rate:
    """The rate of ``outcomes`` over ``jobs``, with the task as the unit.

    The seed is fixed, so the same jobs always give the same interval, in
    whatever order they come.

    The interval is a bootstrap over tasks, as issue #14 asks, and a bootstrap
    cannot see past its sample: where every task has the same rate, every
    resample has it too, and the interval has no width. No silent error in 30
    tasks comes back as 0 with the interval (0, 0), though 30 tasks cannot rule
    out a rate of several percent. A rate at 0 or 1 needs a bound of another
    kind beside it.
    """
    jobs = list(jobs)
    per_task = task_rates(jobs, outcomes)
    if not per_task:
        raise ValueError("no jobs to take a rate over")
    values = np.array(list(per_task.values()), dtype=float)
    low, high = _interval(values, n_boot, confidence, seed)
    return Rate(float(values.mean()), low, high, tasks=int(values.size), jobs=len(jobs))


def paired_difference(
    jobs_a: Iterable[Scored],
    jobs_b: Iterable[Scored],
    outcomes: Collection[Outcome],
    *,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Rate:
    """The rate of ``outcomes`` in ``jobs_a`` less that in ``jobs_b``, over the
    tasks both ran, each task's difference taken first.

    The same resampled tasks serve both sides, so the interval keeps what the
    two shared: a task that is hard for every engine does not widen it. A task
    only one side ran is refused, since the two would then be compared on
    different tasks. ``jobs`` in the result counts both sides.
    """
    jobs_a, jobs_b = list(jobs_a), list(jobs_b)
    outcomes = _outcomes(outcomes)  # once, for both sides
    a, b = task_rates(jobs_a, outcomes), task_rates(jobs_b, outcomes)
    if set(a) != set(b):
        only = sorted(set(a) ^ set(b))
        raise ValueError(f"the two sides ran different tasks: {', '.join(only)}")
    if not a:
        raise ValueError("no jobs to compare")
    diffs = np.array([a[task] - b[task] for task in sorted(a)], dtype=float)
    low, high = _interval(diffs, n_boot, confidence, seed)
    return Rate(
        float(diffs.mean()), low, high, tasks=int(diffs.size), jobs=len(jobs_a) + len(jobs_b)
    )


@dataclass(frozen=True)
class Cell:
    """One row of the outcome table: the jobs of one engine and arm (or of
    whatever ``by`` names), their two rates, and every outcome's count."""

    key: tuple[str, ...]
    silently_wrong: Rate
    completion: Rate
    counts: dict[str, int] = field(default_factory=dict)


def summarize(
    jobs: Iterable[Scored],
    *,
    by: Sequence[str] = ("engine", "arm"),
    exclude_categories: Collection[str] = (),
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> list[Cell]:
    """The silently-wrong and completion rates of each cell, sorted by key.

    ``by`` names the fields of a job that make a cell: ``("engine", "arm")`` for
    the headline table, ``("engine", "arm", "category")`` by category.
    A cell pools whatever ``by`` does not name: with two agents in the study,
    add ``"agent"``, or one agent's jobs are averaged into another's tasks.
    Cells are not held to one set of tasks; each ``Rate`` says how many it has,
    and ``paired_difference`` is what compares engines on the same ones.

    ``exclude_categories`` drops whole categories first. The refusal tasks
    favour an engine that refuses, which is the property under test, so the
    headline is also given without them (issue #12):
    ``exclude_categories=("should_refuse",)``.
    """
    if isinstance(exclude_categories, (str, bytes)):
        raise TypeError("exclude_categories: a collection of categories, not one name")
    exclude_categories = set(exclude_categories)
    cells: dict[tuple[str, ...], list[Scored]] = {}
    for job in jobs:
        if job.category in exclude_categories:
            continue
        cells.setdefault(tuple(str(getattr(job, name)) for name in by), []).append(job)
    kw = {"n_boot": n_boot, "confidence": confidence, "seed": seed}
    return [
        Cell(
            key=key,
            silently_wrong=rate(cell, SILENT, **kw),
            completion=rate(cell, COMPLETE, **kw),
            counts=tally(job.outcome for job in cell),
        )
        for key, cell in sorted(cells.items())
    ]


# --------------------------------------------------------------------------- #
# Causes
# --------------------------------------------------------------------------- #


def uncoded(jobs: Iterable[Scored]) -> list[Scored]:
    """The jobs that need a cause and have none. Every ``silently_wrong``,
    ``unfaithful_answer``, ``wrong_refusal`` and ``failure`` gets one."""
    return [job for job in jobs if Outcome(job.outcome) in NEEDS_CAUSE and job.cause is None]


def agreement(codes_a: Sequence[Cause | str], codes_b: Sequence[Cause | str]) -> dict[str, float]:
    """How far two coders agree on the causes of the same jobs, in order: the
    fraction they agree on, and Cohen's kappa, which takes out what two coders
    choosing at their own rates would agree on by chance.

    On the pilot two people code the causes independently and report this. The
    codes are matched by position: the same jobs, in the same order.

    Kappa is nan where both gave every job the one same cause: chance agreement
    is then total, and kappa is 0/0. The agreement, 1.0, is what there is to
    report.
    """
    if len(codes_a) != len(codes_b):
        raise ValueError("the two coders must code the same jobs")
    if not codes_a:
        raise ValueError("no codes to compare")
    a = [Cause(c) for c in codes_a]
    b = [Cause(c) for c in codes_b]
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    count_a, count_b = Counter(a), Counter(b)
    chance = sum(count_a[c] * count_b[c] for c in Cause) / (n * n)
    kappa = math.nan if chance == 1.0 else (observed - chance) / (1.0 - chance)
    return {"agreement": observed, "kappa": kappa, "jobs": float(n)}


def scored(job: Mapping[str, Any], grade_: Grade, task: _tasks.Task) -> Scored:
    """A graded job as the rates need it, from the job's own axes (issue #15:
    ``engine``, ``arm``, ``agent``, ``repetition``)."""
    return Scored(
        task=task.id,
        category=task.category,
        engine=str(job["engine"]),
        arm=str(job["arm"]),
        agent=str(job["agent"]),
        repetition=int(job["repetition"]),
        outcome=grade_.outcome,
        checked=grade_.checked,
        cause=Cause(job["cause"]) if job.get("cause") else None,
        worst_miss=grade_.worst_miss,
    )
