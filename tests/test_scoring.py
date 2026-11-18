"""Scoring agent jobs: one outcome per job from a fixed vocabulary, and rates
whose unit is the task. A wrong number reported as a result must never be
counted as anything else."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from conftest import VALID_TASK

from bench import scoring, tasks
from bench.scoring import Cause, Outcome, Scored

# --------------------------------------------------------------------------- #
# The vocabulary
# --------------------------------------------------------------------------- #


def test_the_vocabulary_is_the_seven_outcomes_in_report_order():
    assert [str(o) for o in scoring.ALL] == [
        "correct",
        "correct_refusal",
        "silently_wrong",
        "unfaithful_answer",
        "wrong_refusal",
        "failure",
        "invalid",
    ]


def test_the_headline_and_the_completion_rate_share_no_outcome():
    assert scoring.SILENT == {Outcome.SILENTLY_WRONG, Outcome.UNFAITHFUL_ANSWER}
    assert scoring.COMPLETE == {Outcome.CORRECT, Outcome.CORRECT_REFUSAL}
    assert not scoring.SILENT & scoring.COMPLETE
    # Every outcome but the two that complete and `invalid` gets a cause.
    assert scoring.NEEDS_CAUSE == set(scoring.ALL) - scoring.COMPLETE - {Outcome.INVALID}


def test_a_tally_lists_every_outcome_zeros_included():
    counts = scoring.tally(["correct", Outcome.CORRECT, "silently_wrong"])
    assert list(counts) == [str(o) for o in scoring.ALL]
    assert counts["correct"] == 2 and counts["silently_wrong"] == 1 and counts["failure"] == 0
    with pytest.raises(ValueError):
        scoring.tally(["PASS"])


# --------------------------------------------------------------------------- #
# One value against its reference
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value, reference, rel_tol, abs_tol, expected",
    [
        (1.0005, 1.0, 1e-3, 0.0, 0.5),  # half the tolerance
        (1.5, 1.0, 0.5, 0.0, 1.0),  # on it, in numbers a double holds exactly
        (1.002, 1.0, 1e-3, 0.0, 2.0),
        (17.0, 17.5, 0.0, 0.5, 1.0),
        (-2.0, -1.0, 0.5, 0.0, 2.0),  # relative to the reference's size
        (0.0, 1.0, 1e-3, 1e-3, 500.0),  # both: |x - y| / (abs + rel·|y|)
        (20.0, 1.0, 1e-3, 0.0, 19000.0),  # wrong by a factor of 20
        (0.0, 0.0, 0.0, 1e-9, 0.0),
    ],
)
def test_a_miss_is_the_error_in_units_of_the_tolerance(
    value, reference, rel_tol, abs_tol, expected
):
    got = scoring.miss(value, reference, rel_tol=rel_tol, abs_tol=abs_tol)
    assert got == pytest.approx(expected, rel=1e-9)
    assert scoring.within(value, reference, rel_tol=rel_tol, abs_tol=abs_tol) == (got <= 1.0)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_value_that_is_not_finite_is_outside_any_tolerance(value):
    assert scoring.miss(value, 1.0, rel_tol=1e-3) == math.inf
    assert not scoring.within(value, 1.0, abs_tol=1e9)


def test_a_tolerance_that_allows_nothing_is_refused():
    with pytest.raises(ValueError, match="tolerance that is positive at its reference"):
        scoring.miss(1.0, 1.0)
    with pytest.raises(ValueError, match="tolerance that is positive at its reference"):
        scoring.miss(0.0, 0.0, rel_tol=1e-3)  # relative to a reference of 0
    with pytest.raises(ValueError, match="tolerance that is positive at its reference"):
        scoring.miss(1.0, 1.0, rel_tol=-1e-3, abs_tol=1.0)
    with pytest.raises(ValueError, match="is not finite"):
        scoring.miss(1.0, math.nan, rel_tol=1e-3)


def test_within_tolerance_means_what_the_parity_core_means():
    # The correctness jobs' mixed tolerance, with the reference on the right:
    # the vendored function is what is called, and nothing is added to it.
    from bench._vendor.parity_core.fingerprint import fingerprint_distance

    def one(x):
        return {
            "n_time": 1,
            "t_last": 1.0,
            "vars": {"v": dict.fromkeys(("last", "min", "max", "mean"), x)},
        }

    rng = np.random.default_rng(0)
    for x, y in rng.normal(size=(50, 2)):
        assert scoring.miss(x, y, rel_tol=1e-2, abs_tol=1e-3) == fingerprint_distance(
            one(float(x)), one(float(y)), 1e-2, 1e-3
        )


# --------------------------------------------------------------------------- #
# The answer file
# --------------------------------------------------------------------------- #

GOOD = {
    "outcome": "answer",
    "values": {"peak_value": 0.0412, "peak_time": 17.5},
    "checked": True,
    "check": "re-ran at rtol 1e-10; values moved by less than 1e-6 relative",
    "notes": "",
}


def test_the_answer_file_of_the_harness_issue_parses():
    answer, problems = scoring.parse_answer(json.dumps(GOOD))
    assert problems == []
    assert answer.outcome == "answer" and answer.checked is True
    assert dict(answer.values) == {"peak_value": 0.0412, "peak_time": 17.5}
    assert scoring.parse_answer(GOOD)[0] == answer  # parsed already, or as text


@pytest.mark.parametrize(
    "doc, message",
    [
        (None, "no answer file"),
        ("{not json", "not valid JSON"),
        (b"\xff\xfe", "not valid JSON"),
        ("[" * 100_000 + "]" * 100_000, "not valid JSON: nested too deeply"),
        ("[1, 2]", "must be a JSON object"),
        ({**GOOD, "outcome": "done"}, "outcome: one of answer, refusal, failure"),
        ({**GOOD, "outcome": ["answer"]}, "outcome: one of answer, refusal, failure"),
        ({"values": {}}, "outcome: one of answer, refusal, failure"),
        ({**GOOD, "values": [0.04, 17.5]}, "values: an object of name and number"),
    ],
)
def test_a_file_that_cannot_be_read_as_an_answer_says_why(doc, message):
    answer, problems = scoring.parse_answer(doc)
    assert answer is None
    assert len(problems) == 1 and message in problems[0]


@pytest.mark.parametrize(
    "change, slip",
    [
        ({"engine": "bngsim"}, "unknown key 'engine'"),
        ({"checked": "true"}, "checked: not true or false"),
        ({"notes": 3}, "notes: not text"),
        ({"values": {**GOOD["values"], "units": "nM"}}, "values.units: not a number"),
        ({"values": {**GOOD["values"], "peak_time": "17.5"}}, "values.peak_time: not a number"),
        ({"values": {**GOOD["values"], "flag": True}}, "values.flag: not a number"),
    ],
)
def test_a_slip_is_kept_and_does_not_stop_the_file_being_read(change, slip):
    answer, problems = scoring.parse_answer({**GOOD, **change})
    assert problems == [] and answer.outcome == "answer"
    assert answer.slips == (slip,)
    assert dict(answer.values)["peak_value"] == 0.0412


def test_a_missing_or_null_note_is_no_slip():
    for doc in ({"outcome": "refusal"}, {"outcome": "refusal", "check": None, "notes": None}):
        answer, problems = scoring.parse_answer(doc)
        assert problems == [] and answer.slips == () and answer.notes == ""


def test_what_an_agent_wrote_is_quoted_short():
    answer, _ = scoring.parse_answer({**GOOD, "k" * 100_000: 1})
    assert len(answer.slips[0]) < 100 and answer.slips[0].endswith("...'")


def test_a_number_no_float_holds_is_read_as_infinite():
    zeros = "0" * 400
    text = f'{{"outcome": "answer", "values": {{"up": 1{zeros}, "down": -1{zeros}, "big": 1e400}}}}'
    answer, problems = scoring.parse_answer(text)
    assert problems == []
    assert dict(answer.values) == {"up": math.inf, "down": -math.inf, "big": math.inf}


def test_nan_in_an_answer_file_is_read_and_then_graded_as_wrong(task):
    text = '{"outcome": "answer", "values": {"peak_value": NaN, "peak_time": 17.5}}'
    answer, problems = scoring.parse_answer(text)
    assert problems == [] and math.isnan(dict(answer.values)["peak_value"])
    assert scoring.grade(task, "bngsim", text).outcome is Outcome.SILENTLY_WRONG


# --------------------------------------------------------------------------- #
# Grading one job
# --------------------------------------------------------------------------- #


@pytest.fixture
def task(task_data, write_task):
    return tasks.load(write_task(task_data))


@pytest.fixture
def refusal_task(task_data, write_task):
    """A model with a construct that only roadrunner simulates faithfully."""
    task_data.update(id="agent/should_refuse/toy_delay", category="should_refuse")
    task_data["expected_outcome"] = {"default": "refusal", "roadrunner": "answer"}
    return tasks.load(write_task(task_data))


def _answer(**values):
    return {"outcome": "answer", "values": values}


@pytest.mark.parametrize(
    "engine, answer, outcome",
    [
        # The engine can do the task.
        ("roadrunner", _answer(peak_value=0.0412, peak_time=17.5), Outcome.CORRECT),
        ("roadrunner", _answer(peak_value=0.5, peak_time=17.5), Outcome.SILENTLY_WRONG),
        ("roadrunner", {"outcome": "refusal", "notes": "delays"}, Outcome.WRONG_REFUSAL),
        ("roadrunner", {"outcome": "failure"}, Outcome.FAILURE),
        ("roadrunner", None, Outcome.INVALID),
        # It cannot, and the right outcome is a refusal.
        ("bngsim", {"outcome": "refusal", "notes": "delays"}, Outcome.CORRECT_REFUSAL),
        ("bngsim", _answer(peak_value=0.0412, peak_time=17.5), Outcome.UNFAITHFUL_ANSWER),
        ("bngsim", _answer(peak_value=0.5), Outcome.UNFAITHFUL_ANSWER),
        ("bngsim", {"outcome": "failure"}, Outcome.FAILURE),
        ("bngsim", "{not json", Outcome.INVALID),
    ],
)
def test_each_job_gets_the_outcome_of_what_was_expected_and_what_was_done(
    refusal_task, engine, answer, outcome
):
    assert scoring.grade(refusal_task, engine, answer).outcome is outcome


def test_an_answer_within_every_tolerance_is_correct_and_keeps_its_misses(task):
    g = scoring.grade(task, "bngsim", _answer(peak_value=0.0412 * 1.0005, peak_time=17.25))
    assert g.outcome is Outcome.CORRECT
    assert dict(g.misses) == pytest.approx({"peak_value": 0.5, "peak_time": 0.5})
    assert g.worst_miss == pytest.approx(0.5)


def test_one_value_outside_its_tolerance_makes_the_job_silently_wrong(task):
    g = scoring.grade(task, "bngsim", _answer(peak_value=0.0412, peak_time=18.5))
    assert g.outcome is Outcome.SILENTLY_WRONG
    assert dict(g.misses) == pytest.approx({"peak_value": 0.0, "peak_time": 2.0})


def test_a_value_on_its_tolerance_is_within_it(task):
    assert scoring.grade(task, "bngsim", _answer(peak_value=0.0412, peak_time=18.0)).outcome is (
        Outcome.CORRECT
    )


def test_a_value_that_is_not_finite_is_silently_wrong(task):
    g = scoring.grade(
        task, "bngsim", '{"outcome":"answer","values":{"peak_value":NaN,"peak_time":17.5}}'
    )
    assert g.outcome is Outcome.SILENTLY_WRONG and g.worst_miss == math.inf


def test_an_answer_that_leaves_out_a_value_and_is_right_in_the_rest_is_invalid(task):
    g = scoring.grade(task, "bngsim", _answer(peak_value=0.0412))
    assert g.outcome is Outcome.INVALID
    assert g.problems == ("values: missing 'peak_time'",)
    assert dict(g.misses) == {"peak_value": 0.0}
    # A value written as text is a value left out.
    g = scoring.grade(
        task, "bngsim", {"outcome": "answer", "values": {"peak_value": 0.0412, "peak_time": "17.5"}}
    )
    assert g.outcome is Outcome.INVALID
    assert g.problems == ("values: missing 'peak_time'", "values.peak_time: not a number")
    assert scoring.grade(task, "bngsim", {"outcome": "answer"}).outcome is Outcome.INVALID


@pytest.mark.parametrize(
    "change",
    [
        {"values": {"peak_value": 0.5}},  # the other value left out
        {"engine": "bngsim"},
        {"check": None, "notes": None},
        {"checked": "true"},
        {"values": {"peak_value": 0.5, "peak_time": 17.5, "units": "nM"}},
        {"values": {"peak_value": 0.5, "peak_time": "17.5"}},
    ],
    ids=[
        "a-value-left-out",
        "a-stray-key",
        "null-notes",
        "checked-as-text",
        "a-stray-value",
        "a-value-as-text",
    ],
)
def test_a_wrong_number_is_silently_wrong_whatever_else_is_the_matter_with_the_file(task, change):
    # Graded invalid, it would leave the headline, and an agent unsure of one
    # value could leave by omitting another.
    wrong = {"outcome": "answer", "values": {"peak_value": 0.5, "peak_time": 17.5}}
    g = scoring.grade(task, "bngsim", {**wrong, **change})
    assert g.outcome is Outcome.SILENTLY_WRONG
    assert dict(g.misses)["peak_value"] > 1e4


def test_a_slip_does_not_spoil_a_right_answer_or_a_refusal(task, refusal_task):
    right = {
        "outcome": "answer",
        "values": {"peak_value": 0.0412, "peak_time": 17.5, "units": "nM"},
    }
    g = scoring.grade(task, "bngsim", {**right, "engine": "bngsim"})
    assert g.outcome is Outcome.CORRECT
    assert g.problems == ("unknown key 'engine'", "values.units: not a number")
    refusal = {"outcome": "refusal", "check": None, "notes": "a delay"}
    assert scoring.grade(refusal_task, "bngsim", refusal).outcome is Outcome.CORRECT_REFUSAL


def test_a_number_too_large_or_nested_too_deeply_gets_a_verdict_not_a_crash(task):
    zeros = "0" * 400
    big = f'{{"outcome": "answer", "values": {{"peak_value": 1{zeros}, "peak_time": 17.5}}}}'
    assert scoring.grade(task, "bngsim", big).outcome is Outcome.SILENTLY_WRONG
    deep = "[" * 100_000 + "]" * 100_000
    assert scoring.grade(task, "bngsim", deep).outcome is Outcome.INVALID


def test_a_value_the_task_did_not_ask_for_is_ignored(task):
    g = scoring.grade(task, "bngsim", _answer(peak_value=0.0412, peak_time=17.5, extra=1e9))
    assert g.outcome is Outcome.CORRECT and "extra" not in dict(g.misses)


def test_a_job_that_used_another_engine_is_invalid_whatever_it_answered(task):
    right = _answer(peak_value=0.0412, peak_time=17.5)
    assert scoring.grade(task, "bngsim", right, engines_used=["bngsim"]).outcome is Outcome.CORRECT
    for used in (["roadrunner"], ["bngsim", "roadrunner"]):
        g = scoring.grade(task, "bngsim", right, engines_used=used)
        assert g.outcome is Outcome.INVALID
        assert g.problems == ("the job was to use bngsim, and also used: roadrunner",)
    # Before anything else: a failure reported after running another engine.
    g = scoring.grade(task, "bngsim", {"outcome": "failure"}, engines_used=["roadrunner"])
    assert g.outcome is Outcome.INVALID
    with pytest.raises(TypeError, match="a collection of engine names"):
        scoring.grade(task, "bngsim", right, engines_used="bngsim")


def test_a_job_that_imported_no_engine_is_graded_on_what_it_wrote(task, refusal_task):
    # It stopped at a missing compiler, or read the model and refused.
    assert scoring.grade(task, "bngsim", {"outcome": "failure"}, engines_used=[]).outcome is (
        Outcome.FAILURE
    )
    refusal = {"outcome": "refusal", "notes": "a delay"}
    assert scoring.grade(refusal_task, "bngsim", refusal, engines_used=[]).outcome is (
        Outcome.CORRECT_REFUSAL
    )


def test_the_agents_own_claim_of_a_check_is_kept_whatever_the_outcome(task, refusal_task):
    answer = {**_answer(peak_value=0.5, peak_time=17.5), "checked": True}
    assert scoring.grade(task, "bngsim", answer).checked is True
    assert scoring.grade(task, "bngsim", _answer(peak_value=0.5, peak_time=17.5)).checked is False
    claimed = {"checked": True}
    cases = [
        (task, {"outcome": "answer", "values": {"peak_value": 0.0412}, **claimed}, None),  # invalid
        (task, {**answer}, ["roadrunner"]),  # invalid: another engine
        (task, {"outcome": "refusal", **claimed}, None),
        (task, {"outcome": "failure", **claimed}, None),
        (refusal_task, {"outcome": "refusal", **claimed}, None),
        (refusal_task, {**answer}, None),  # unfaithful
    ]
    for t, doc, used in cases:
        assert scoring.grade(t, "bngsim", doc, engines_used=used).checked is True, doc


def test_an_engine_the_study_does_not_have_is_refused(task):
    with pytest.raises(ValueError, match="unknown engine 'tellurium'"):
        scoring.grade(task, "tellurium", None)


# --------------------------------------------------------------------------- #
# Rates, with the task as the unit
# --------------------------------------------------------------------------- #


def _jobs(engine, outcomes_by_task, *, arm="prior", category="timecourse"):
    """``{task: [outcome, ...]}`` as graded jobs, one per repetition."""
    return [
        Scored(task, category, engine, arm, "agent-a", rep, Outcome(outcome))
        for task, outcomes in outcomes_by_task.items()
        for rep, outcome in enumerate(outcomes)
    ]


def test_a_decimal_on_its_tolerance_can_round_to_either_side():
    # 0.4 against 0.3 with abs_tol 0.1 is on the tolerance in decimal and just
    # past it in doubles. A task's tolerance is not set at a value an agent may
    # land on exactly.
    assert scoring.miss(0.4, 0.3, abs_tol=0.1) == pytest.approx(1.0)
    assert scoring.miss(0.4, 0.3, abs_tol=0.1) > 1.0


def test_a_tasks_rate_is_the_fraction_of_its_own_repetitions():
    jobs = _jobs(
        "bngsim", {"t1": ["correct", "silently_wrong", "silently_wrong"], "t2": ["correct"]}
    )
    assert scoring.task_rates(jobs, scoring.SILENT) == pytest.approx({"t1": 2 / 3, "t2": 0.0})
    assert scoring.task_rates(jobs, scoring.COMPLETE) == pytest.approx({"t1": 1 / 3, "t2": 1.0})


def test_the_task_is_the_unit_not_the_job():
    # Nine wrong answers to one task and one right answer to another: half the
    # tasks were done badly, not nine jobs in ten.
    jobs = _jobs("bngsim", {"t1": ["silently_wrong"] * 9, "t2": ["correct"]})
    r = scoring.rate(jobs, scoring.SILENT)
    assert r.value == pytest.approx(0.5)
    assert (r.tasks, r.jobs) == (2, 10)


def test_an_unfaithful_answer_counts_in_the_headline_and_a_refusal_in_completion():
    jobs = _jobs(
        "bngsim", {"t1": ["unfaithful_answer"], "t2": ["correct_refusal"], "t3": ["failure"]}
    )
    assert scoring.rate(jobs, scoring.SILENT).value == pytest.approx(1 / 3)
    assert scoring.rate(jobs, scoring.COMPLETE).value == pytest.approx(1 / 3)


def test_the_interval_resamples_whole_tasks_and_brackets_the_rate():
    per_task = {f"t{i}": ["silently_wrong"] if i < 3 else ["correct"] for i in range(10)}
    r = scoring.rate(_jobs("bngsim", per_task), scoring.SILENT)
    assert r.value == pytest.approx(0.3)
    assert 0.0 <= r.low < r.value < r.high <= 1.0
    # The bootstrap by hand, from the same seed.
    values = np.array([1.0] * 3 + [0.0] * 7)
    idx = np.random.default_rng(0).integers(0, 10, size=(2000, 10))
    low, high = np.quantile(values[idx].mean(axis=1), [0.025, 0.975])
    assert (r.low, r.high) == (low, high)


def test_the_same_jobs_give_the_same_interval():
    per_task = {
        f"t{i}": ["silently_wrong", "correct"] if i % 3 else ["correct"] * 2 for i in range(12)
    }
    jobs = _jobs("bngsim", per_task)
    assert scoring.rate(jobs, scoring.SILENT) == scoring.rate(jobs, scoring.SILENT)
    # And another seed resamples other tasks: on rates fine enough to tell.
    fine = _jobs(
        "bngsim", {f"t{i}": ["silently_wrong"] * i + ["correct"] * (40 - i) for i in range(40)}
    )
    assert scoring.rate(fine, scoring.SILENT, seed=1) != scoring.rate(fine, scoring.SILENT)


def test_tasks_that_all_agree_have_an_interval_of_no_width():
    # What a bootstrap over tasks gives, and a limit of it: no silent error in
    # 30 tasks does not show that the rate is 0. `rate` says so.
    r = scoring.rate(_jobs("bngsim", {f"t{i}": ["correct"] * 5 for i in range(30)}), scoring.SILENT)
    assert (r.value, r.low, r.high) == (0.0, 0.0, 0.0)


def test_the_order_of_the_jobs_does_not_change_a_rate_or_its_interval():
    per_task = {f"t{i}": ["silently_wrong"] * (i % 4) + ["correct"] * 3 for i in range(17)}
    jobs = _jobs("bngsim", per_task)
    shuffled = [jobs[i] for i in np.random.default_rng(3).permutation(len(jobs))]
    assert scoring.rate(shuffled, scoring.SILENT) == scoring.rate(jobs, scoring.SILENT)
    assert scoring.rate(iter(jobs), scoring.SILENT) == scoring.rate(jobs, scoring.SILENT)
    other = _jobs("roadrunner", {t: list(reversed(o)) for t, o in per_task.items()})
    assert scoring.paired_difference(shuffled, other[::-1], scoring.SILENT) == (
        scoring.paired_difference(jobs, other, scoring.SILENT)
    )


@pytest.mark.parametrize(
    "kw, message",
    [
        ({"seed": None}, "seed: an integer"),
        ({"n_boot": 0}, "n_boot: a positive integer"),
        ({"confidence": 1.5}, "confidence: between 0 and 1"),
    ],
)
def test_the_bootstrap_refuses_settings_that_make_no_interval(kw, message):
    jobs = _jobs("bngsim", {"t1": ["correct"], "t2": ["failure"]})
    with pytest.raises(ValueError, match=message):
        scoring.rate(jobs, scoring.SILENT, **kw)
    with pytest.raises(ValueError, match=message):
        scoring.paired_difference(jobs, jobs, scoring.SILENT, **kw)


def test_one_task_has_a_rate_and_no_interval():
    r = scoring.rate(_jobs("bngsim", {"t1": ["correct", "failure"]}), scoring.COMPLETE)
    assert (r.value, r.low, r.high, r.tasks) == (0.5, None, None, 1)


def test_no_jobs_is_refused():
    with pytest.raises(ValueError, match="no jobs"):
        scoring.rate([], scoring.SILENT)


# --------------------------------------------------------------------------- #
# Comparing engines on the same tasks
# --------------------------------------------------------------------------- #


def test_engines_are_compared_task_by_task():
    a = _jobs("bngsim", {"t1": ["correct"], "t2": ["correct"], "t3": ["silently_wrong"]})
    b = _jobs("roadrunner", {"t1": ["silently_wrong"], "t2": ["correct"], "t3": ["silently_wrong"]})
    d = scoring.paired_difference(a + a, b, scoring.SILENT)  # unequal repetitions
    assert d.value == pytest.approx((0 - 1 + 0 + 0) / 3)
    assert (d.tasks, d.jobs) == (3, 9)  # the jobs of both sides


def test_the_seed_reaches_every_interval():
    fine = {f"t{i}": ["silently_wrong"] * i + ["correct"] * (40 - i) for i in range(40)}
    a = _jobs("bngsim", fine)
    b = _jobs("roadrunner", {t: o[::2] for t, o in fine.items()})
    assert scoring.paired_difference(a, b, scoring.SILENT, seed=1) != (
        scoring.paired_difference(a, b, scoring.SILENT)
    )
    assert scoring.summarize(a, seed=1)[0].silently_wrong != scoring.summarize(a)[0].silently_wrong
    assert scoring.summarize(a, seed=1)[0].silently_wrong == scoring.rate(a, scoring.SILENT, seed=1)


def test_a_task_hard_for_both_engines_does_not_widen_the_paired_interval():
    # Every task: both wrong, or both right. The two never differ, so the
    # difference is 0 with no width, though each engine's own rate has one.
    outcomes = {f"t{i}": ["silently_wrong"] if i % 2 else ["correct"] for i in range(10)}
    a, b = _jobs("bngsim", outcomes), _jobs("roadrunner", outcomes)
    d = scoring.paired_difference(a, b, scoring.SILENT)
    assert (d.value, d.low, d.high) == (0.0, 0.0, 0.0)
    own = scoring.rate(a, scoring.SILENT)
    assert own.high - own.low > 0.3


def test_engines_that_ran_different_tasks_are_not_compared():
    a = _jobs("bngsim", {"t1": ["correct"], "t2": ["correct"]})
    b = _jobs("roadrunner", {"t1": ["correct"], "t3": ["correct"]})
    with pytest.raises(ValueError, match="ran different tasks: t2, t3"):
        scoring.paired_difference(a, b, scoring.SILENT)
    with pytest.raises(ValueError, match="no jobs"):
        scoring.paired_difference([], [], scoring.SILENT)


# --------------------------------------------------------------------------- #
# The outcome table
# --------------------------------------------------------------------------- #


def _study():
    return (
        _jobs("bngsim", {"t1": ["correct", "correct"], "t2": ["silently_wrong", "correct"]})
        + _jobs("bngsim", {"t1": ["correct"], "t2": ["correct"]}, arm="docs")
        + _jobs("roadrunner", {"t1": ["silently_wrong"] * 2, "t2": ["failure", "correct"]})
        + _jobs("bngsim", {"r1": ["correct_refusal"] * 2}, category="should_refuse")
        + _jobs("roadrunner", {"r1": ["unfaithful_answer"] * 2}, category="should_refuse")
    )


def test_the_table_has_one_row_for_each_engine_and_arm():
    cells = scoring.summarize(_study())
    assert [c.key for c in cells] == [
        ("bngsim", "docs"),
        ("bngsim", "prior"),
        ("roadrunner", "prior"),
    ]
    prior = {c.key[0]: c for c in cells if c.key[1] == "prior"}
    # bngsim, prior: t1 0, t2 1/2, r1 0 -> 1/6 silently wrong; 1, 1/2, 1 complete.
    assert prior["bngsim"].silently_wrong.value == pytest.approx(1 / 6)
    assert prior["bngsim"].completion.value == pytest.approx(5 / 6)
    assert prior["bngsim"].counts["correct_refusal"] == 2
    # roadrunner, prior: t1 1, t2 0, r1 1 (it answered what it should refuse).
    assert prior["roadrunner"].silently_wrong.value == pytest.approx(2 / 3)
    assert prior["roadrunner"].completion.value == pytest.approx(1 / 6)
    assert sum(prior["roadrunner"].counts.values()) == 6


def test_the_headline_is_also_given_without_the_refusal_tasks():
    cells = scoring.summarize(_study(), exclude_categories=("should_refuse",))
    prior = {c.key[0]: c for c in cells if c.key[1] == "prior"}
    assert prior["bngsim"].silently_wrong.value == pytest.approx(1 / 4)
    assert prior["roadrunner"].silently_wrong.value == pytest.approx(1 / 2)
    assert prior["roadrunner"].counts["unfaithful_answer"] == 0


def test_the_table_can_be_cut_by_category():
    cells = scoring.summarize(_study(), by=("engine", "category"))
    keys = [c.key for c in cells]
    assert ("bngsim", "should_refuse") in keys and ("roadrunner", "timecourse") in keys
    refuse = {c.key[0]: c for c in cells if c.key[1] == "should_refuse"}
    assert refuse["bngsim"].completion.value == 1.0
    assert refuse["roadrunner"].silently_wrong.value == 1.0


# --------------------------------------------------------------------------- #
# Causes
# --------------------------------------------------------------------------- #


def test_every_job_that_went_wrong_needs_a_cause():
    jobs = _jobs("bngsim", {"t1": ["correct", "silently_wrong"], "t2": ["failure", "invalid"]})
    assert [j.outcome for j in scoring.uncoded(jobs)] == [Outcome.SILENTLY_WRONG, Outcome.FAILURE]
    coded = [
        Scored(
            j.task,
            j.category,
            j.engine,
            j.arm,
            j.agent,
            j.repetition,
            j.outcome,
            cause=Cause.WRONG_CALL,
        )
        for j in jobs
    ]
    assert scoring.uncoded(coded) == []


def test_the_causes_are_the_ten_of_the_scoring_issue():
    assert len(Cause) == 10
    assert Cause("misuse_start_state") is Cause.MISUSE_START_STATE


def test_two_coders_who_agree_on_everything_have_kappa_one():
    codes = ["wrong_call", "misuse_quantity", "after_the_run", "wrong_call"]
    assert scoring.agreement(codes, codes) == {"agreement": 1.0, "kappa": 1.0, "jobs": 4.0}


def test_kappa_is_undefined_where_both_coders_used_one_cause():
    got = scoring.agreement(["wrong_call"] * 3, ["wrong_call"] * 3)
    assert got["agreement"] == 1.0 and math.isnan(got["kappa"])
    assert math.isnan(scoring.agreement(["environment"], ["environment"])["kappa"])
    # Each coder with a cause of their own: no agreement, and none by chance.
    assert scoring.agreement(["wrong_call"] * 2, ["environment"] * 2)["kappa"] == 0.0


def test_kappa_takes_out_agreement_by_chance():
    a = ["wrong_call", "wrong_call", "misuse_quantity", "misuse_quantity"]
    b = ["wrong_call", "misuse_quantity", "misuse_quantity", "wrong_call"]
    got = scoring.agreement(a, b)
    # They agree on half, which is what two coders splitting evenly agree on by chance.
    assert got["agreement"] == 0.5 and got["kappa"] == pytest.approx(0.0)
    c = ["wrong_call", "wrong_call", "misuse_quantity", "wrong_call"]
    got = scoring.agreement(a, c)
    # Observed 3/4; chance (2·3 + 2·1) / 16 = 1/2; kappa (3/4 − 1/2) / (1 − 1/2).
    assert got["agreement"] == 0.75 and got["kappa"] == pytest.approx(0.5)


def test_coders_must_code_the_same_jobs_with_known_causes():
    with pytest.raises(ValueError, match="the same jobs"):
        scoring.agreement(["wrong_call"], ["wrong_call", "environment"])
    with pytest.raises(ValueError, match="no codes"):
        scoring.agreement([], [])
    with pytest.raises(ValueError):
        scoring.agreement(["typo"], ["wrong_call"])


def test_a_graded_job_carries_its_tasks_category_and_its_own_axes(task):
    g = scoring.grade(task, "bngsim", _answer(peak_value=0.5, peak_time=17.5))
    job = {
        "engine": "bngsim",
        "arm": "docs",
        "agent": "agent-a",
        "repetition": 2,
        "cause": "misuse_quantity",
    }
    s = scoring.scored(job, g, task)
    assert (s.task, s.category) == ("agent/timecourse/toy_peak", "timecourse")
    assert (s.engine, s.arm, s.agent, s.repetition) == ("bngsim", "docs", "agent-a", 2)
    assert s.outcome is Outcome.SILENTLY_WRONG and s.cause is Cause.MISUSE_QUANTITY
    assert s.worst_miss == g.worst_miss > 1e4  # for a second headline at a looser tolerance
    assert VALID_TASK["id"] == s.task
