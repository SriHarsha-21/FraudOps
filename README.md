# FraudOps

Conformal fraud detection with a bounded agent loop, human review and a red team.
Team: Conformal Sentinels.

Status: stage 1 (data, detector, conformal escalation), stage 2 (agent workflow with
human review and a tamper-evident decision store) and stage 3 (backtest) are implemented
and tested on a synthetic fixture. Real-data results are in the Results section below.
Stage 4 (a rule-based red team, a blue retrain and a promotion rule) is implemented and
tested on the fixture. Stage 5 (a Streamlit review app for escalated cases) is implemented.

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

## Review app

    pip install streamlit
    streamlit run app/streamlit_app.py

The app lists escalated cases by amount times fraud probability. It shows the evidence the
agent gathered and its recommendation. A reviewer approves or blocks; only that click puts
a human verdict in the decision record. Pending cases survive a restart (state folder
review_state/). Set FRAUDOPS_REVIEW_PIN to require a shared PIN. The reviewer name is a
label, not an identity check. The card number is never shown, only a short hash reference.
A demo button simulates a reviewer. Agreement figures from it say nothing about real analysts.

## Docker

    docker build -t fraudops .
    docker run -p 8501:8501 -e FRAUDOPS_REVIEW_PIN=1234 fraudops

Continuous integration runs the test suite on every push (.github/workflows/ci.yml).

## Layout

    src/fraudops/          data, features, detector, conformal, pipeline, backtest, sweep, redteam, review
    app/                   streamlit_app.py (review front end)
    src/fraudops/agents/   tools, planner, store, LangGraph workflow
    scripts/               run_stage1.py, run_backtest.py
    tests/                 stage 1 to 5 tests
    config.json     seeds, alpha, detector settings, cost assumptions

## Agents

- Investigation agent (LangGraph): score, plan, act (four evidence tools in a bounded loop), recommend, verify, human review, finalize. Only escalated cases enter the loop.
- Planner: rule-based by default. An optional Llama planner (Ollama) sits behind the same interface. Code validates every planner answer and falls back to the rules when the model fails.
- Explainer: writes a short summary for the reviewer from numbers and tool notes only. It never sets a verdict.
- Verifier: checks that the suggestion follows from the evidence. It can only add flags. A verifier error becomes a flag.
- Drift monitor: a conformal test martingale on the stream of model scores. On an alarm it can tighten the gate to cautious mode. Only a named human can relax it.
- Authority: only the conformal gate or a human sets a verdict. No LLM output does.

## Results

Data: simulated card transactions (Sparkov style) from a public mirror of fraudTrain.csv (dazzle-nu/CIS435-CreditCardFraudDetection). The mirror was saved from a spreadsheet and holds 14 months (2019-01-01 to 2020-03-10) because it stops at the spreadsheet row limit. We kept 200 cards and split by time 70/15/15. The test period has 33,893 transactions with 145 frauds. All numbers come from one seed. Costs use assumed prices: 25 per false block and 5 per review plus the missed fraud amount.

### Detector and conformal gate

| Measure | Value |
| --- | --- |
| Rows kept (200 cards) | train 158169, calibration 33893, test 33893 |
| Fraud rate | train 0.64 percent, calibration 0.54 percent, test 0.43 percent |
| ROC AUC / PR AUC (test) | 0.9996 / 0.957 |
| Calibration error (ECE, test) | 0.0002 after temperature scaling (T = 1.33) |
| Gate coverage at alpha 0.1 | legit 0.915, fraud 0.910 |
| Escalation rate | 8.5 percent |
| Accuracy on auto-decided cases | 0.9995 |

The near-perfect AUC shows that the simulated fraud is easy to separate. A real portfolio would be harder.

### Backtest (test period)

| System | Precision | Recall | Missed fraud amount | False blocks | Reviews | Cost |
| --- | --- | --- | --- | --- | --- | --- |
| Rules only | 0.091 | 0.441 | 23589 | 639 | 0 | 39564 |
| Model only | 0.485 | 0.979 | 1320 | 151 | 0 | 5095 |
| Model plus conformal gate plus human | 0.485 | 1.000 | 0 | 154 | 2878 | 18240 |

The model beats the rules (McNemar p below 1e-90). The gate reaches full recall but costs more on clean data because reviews cost money. The sweep shows the gate is cheaper than a plain model threshold only when a review costs 0.25 (4993 against 5095). At a review price of 5 the model-only threshold is far cheaper (5095 against 10926 at the chosen alpha of 0.02).

The agent resolved 15.8 percent of the 2878 escalated cases (about 454). On those cases it was 99.6 percent accurate against 96.5 percent for the model. Sending every escalation to a human costs 17940 and letting the agent resolve its confident cases costs 15118. The model-only threshold costs 4719.

### Red team (value recall, share of fraud money caught)

| Attack | Model | Gate | Gate after retrain | Gate fraud coverage | Coverage after retrain |
| --- | --- | --- | --- | --- | --- |
| clean | 0.983 | 0.996 | 0.997 | 0.910 | 0.938 |
| amount shaving | 0.098 | 0.748 | 0.995 | 0.016 | 0.974 |
| splitting | 0.549 | 0.844 | 0.981 | 0.290 | 0.841 |
| timing spread | 0.949 | 0.993 | 0.995 | 0.814 | 0.862 |
| category swap | 0.934 | 0.987 | 1.000 | 0.593 | 0.972 |
| local looking | 0.983 | 0.997 | 1.000 | 0.924 | 0.986 |
| combined | 0.720 | 0.895 | 0.999 | 0.285 | 0.940 |

Attacks keep the total value in cents and only change how the fraud is spread. Amount shaving makes the model miss almost everything. The gate catches much more by escalating. Retraining on mutated fraud recovers the attack families it has seen and passed the promotion rule (clean cost 17404 against 18663 and coverage 0.938).

### Generalization of the retrain (gate value recall)

| Test | Attack | Original gate | Retrained gate |
| --- | --- | --- | --- |
| shifted attack settings | amount shaving | 0.677 | 0.975 |
| shifted attack settings | splitting | 0.340 | 0.870 |
| attack withheld from retraining | amount shaving | 0.748 | 0.654 |
| attack withheld from retraining | splitting | 0.844 | 0.894 |
| attack withheld from retraining | category swap | 0.987 | 0.988 |

Retraining transfers to shifted settings of seen attacks. It does not transfer to attack types it never saw. With amount shaving withheld the retrained gate was worse than the original (0.654 against 0.748).

### Drift monitor

On exchangeable calibration data the monitor raised a false alarm in 2 of 100 runs (bound 1 percent). When 2 percent of later cases became fraud-like it detected the surge in 84 percent of runs after a median of about 5100 cases. At 5 percent and 10 percent it detected 100 percent of runs after about 1750 and 860 cases. On the real test stream it alarmed at case 6017 of 33893 after the fraud rate rose from 0.28 to 0.62 percent. That rise is ordinary variation in this data and not an attack. Under cautious mode the alarm would have sent 25581 automatic approvals to a person, so the monitor needs recalibration after a human acknowledges an alarm.

### Limitations

- The data is simulated and covers 200 cards. The test period has only 145 frauds, so every figure has wide uncertainty.
- One seed. No confidence intervals.
- Attacks are rule-based and keep the total value. Attacked rows are pieces of the same 145 cases.
- The gate coverage guarantee holds for exchangeable data only. Under attack coverage fell to 0.016. Treat the gate as an escalation trigger and not as a guarantee.
- Prices are assumptions. The reviewer is simulated with a 5 percent error rate. No real analysts were involved.
- The agent accuracy figure covers about 454 resolved cases.
- The Llama planner was compared with the rule planner on a synthetic fixture only (19 of 20 recommendations matched).
- The drift monitor alarms on ordinary drift and cannot tell drift from an attack.

## Licence

MIT.
