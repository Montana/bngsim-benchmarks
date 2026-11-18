# Agent tasks

One YAML file per task, read by `bench.tasks` and graded by `bench.scoring`.
A task is what an agent is asked to do with one of the suite's problems: a
request in plain language in place of the problem's protocol, and the named
quantities to report, each with a stored reference and a tolerance (issue #13).

Tasks live here, not under `problems/`: every YAML file there is read as a
problem spec.

```yaml
# tasks/timecourse/kinetic_proofreading_peak.yaml
id: agent/timecourse/kinetic_proofreading_peak     # agent/<category>/<file name>
category: timecourse
model: {problem: kinetic_proofreading_hlavacek2001}   # a problem spec's id
forms: [sbml, bngl]                                   # the files an engine may be handed
request: >
  Simulate the model from its initial state for 100 seconds.
  Report the peak value of <observable> and the time at which it occurs.
answer:
  - {name: peak_value, reference: <value>, rel_tol: 1.0e-3}
  - {name: peak_time,  reference: <value>, abs_tol: <value>}
expected_outcome: {default: answer}     # or, for an engine: answer | refusal
trap: none                              # or the name of a trap, with trap_source
split: public                           # held_out tasks are kept out of this repository
```

| key | what it holds |
|---|---|
| `category` | `timecourse`, `scan`, `steady_state`, `stochastic`, `sensitivity`, `perturbation`, `should_refuse` |
| `answer` | The values to report. Each is within tolerance when `\|value - reference\| <= abs_tol + rel_tol * \|reference\|`; give `rel_tol`, `abs_tol` or both. A reference of 0 needs an `abs_tol`. |
| `expected_outcome` | `default`, and any engine whose right outcome differs: `bngsim`, `roadrunner`, `amici`, `copasi`, `bionetgen`. A `refusal` is expected only in `should_refuse`, and a `should_refuse` task expects one from some engine. |
| `trap` | `none`, or `state_carried_between_runs`, `amount_or_concentration`, `output_grid`, `one_stochastic_run`, `default_tolerance_too_loose`, `not_yet_at_steady_state`, `construct_dropped`. A trap gives `trap_source`: the issue or the documentation that records it. |
| `split` | `public` or `held_out`. Loading this directory refuses a held-out task: they are written with the public ones and kept elsewhere until the results are published (issue #12). |

The request names no engine, no API and no pitfall.

No task is committed yet. A reference is admitted only when two independent
engines agree on it within the suite's correctness tolerances, or an analytic
value exists (issue #13), and the suite stores no references so far.
