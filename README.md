# FraudOps

Conformal fraud detection with a bounded agent loop, human review and a red team.
Team: Conformal Sentinels.

Status: stage 1 (data, detector, conformal escalation), stage 2 (agent workflow with
human review and a tamper-evident decision store) and stage 3 (backtest) are implemented
and tested on a synthetic fixture. Real-data results are pending the first Colab run.
Stage 4 (a rule-based red team, a blue retrain and a promotion rule) is implemented and
tested on the fixture. The review app is planned.

## Originality and AI assistance

All code in this repository was written for this project. No code was copied from
other users' repositories or notebooks. Public pages were read only to learn facts
about the dataset. The code was developed with AI assistance (Claude) and reviewed
by the author. Dependencies are standard open-source libraries.

## Data

Sparkov-style simulated card transactions (fraudTrain.csv and fraudTest.csv, Kaggle
dataset kartik2112/fraud-detection). Data files are never committed. Download them
in Colab or locally and pass the paths on the command line.

The data is simulated, so results describe this benchmark and not real-world fraud.

## Quickstart

    pip install -r requirements.txt
    pytest -q
    python scripts/run_stage1.py                       # dry run on a synthetic fixture
    python scripts/run_stage1.py --train fraudTrain.csv --test fraudTest.csv --max-cards 200

Results are written to results/stage1_metrics.json.

    python scripts/run_backtest.py --train fraudTrain.csv --max-cards 200
    python scripts/run_sweep.py --train fraudTrain.csv --max-cards 200

    python scripts/run_redteam.py --train fraudTrain.csv --max-cards 200

The red team rewrites test-period fraud so each case looks more normal while the total
value taken stays the same, then measures how much recall each system loses. The rules
are in src/fraudops/redteam.py. Robustness is measured against this attacker only. scripts/run_generalization.py checks
the blue retrain against attacks it has not seen: the same attacks with different settings,
and each attack held out of the retraining data.

The sweep picks the conformal alpha on the calibration split only and prints test cost
for several review prices, so the cost of escalation can be read honestly.

## What stage 1 does

1. Validates the schema, cleans the data, drops identity fields and sanitizes text.
2. Builds features that use only earlier transactions of the same card.
3. Splits by time into train, calibration and test.
4. Trains a gradient-boosted detector and applies temperature scaling.
5. Builds class-conditional conformal sets: one label acts, anything else escalates.
6. Reports coverage per class, escalation rate, calibration error and cost.

Gender and state are never used as detector features.

## What stage 2 does

A LangGraph workflow runs each case: score, conformal gate, then either an automatic
action or a bounded investigation (at most agent.max_steps tool calls) and a pause for a
human reviewer. Rules enforced in code:

- A planner can only name an unused tool or say done. Anything else is rejected.
- Only the conformal gate or a human sets a verdict. No LLM output does.
- Escalated cases pause at an interrupt. The checkpointer is SQLite, so a pending
  review survives a restart.
- Every decision is stored in an append-only SQLite table. Each record carries the hash
  of the previous one, so edits are detected by verify_chain().
- A tool error or a trust level of cautious or manual-only widens escalation.

The default planner is rule based and needs no API key. An optional planner that asks
Claude for the next tool is in agents/planner.py. It sends tool names and numbers only
and is not covered by the tests because it needs a key and network access.

## What stage 3 does

    python scripts/run_backtest.py --train fraudTrain.csv --test fraudTest.csv --max-cards 200

Full held-out period: rules only, model only (cost-optimal threshold chosen on the
calibration split) and model plus conformal with a simulated human reviewer, with
McNemar tests between systems. Escalated cases are then run through the agent workflow
(up to backtest.agent_cap of them) to measure how often the agents resolve a case and
whether their recommendation beats the model on the same cases.

Limit: the simulated reviewer returns the true decision with a fixed error rate, so
this backtest cannot show that evidence helps a human. It measures the agents'
recommendation accuracy and the share of cases they resolve instead.

## Assumptions to review

Review cost, false-block cost, the label delay, the reviewer error rate, the tool
scales (tools.*) and the decision thresholds (agent.block_at and agent.approve_at) in
config.json are assumptions. They are starting values to be revisited after the first
real-data baseline.

## Layout

    src/fraudops/          data, features, detector, conformal, pipeline, backtest
    src/fraudops/agents/   tools, planner, store, LangGraph workflow
    scripts/               run_stage1.py, run_backtest.py
    tests/                 stage 1, stage 2 and backtest tests
    config.json     seeds, alpha, detector settings, cost assumptions

## Licence

MIT.
