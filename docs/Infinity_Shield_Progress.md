# Infinity Shield — Progress Tracker

**Whole-project completion: ~30%**
Target: finish everything (thesis + prototype) in 6 weeks, part-time alongside courses.
Reality check: achievable ONLY if O4 hardware is ordered in week 1 and thesis gets protected weekly hours. If hardware slips, realistic total is 8–10 weeks.

How to use this: every time you sit down, open this file first. Tick what's done, read the "current step" line, and you'll be back in flow in two minutes instead of twenty.

---

## CURRENT STEP (update this line every session)
> Writing `test_ekf.py` — fixing the comparison block to use Euclidean distance per timestep. EKF is written but has NEVER been run successfully. First goal: get it tracking a CLEAN trajectory.

---

## O1 — Detection benchmarking + XAI  ✅ 100%
- [x] Everything. Done, documented, on GitHub.

---

## O2 — Trajectory prediction  (~20%)
Goal: EKF, physics-ODE, (maybe LSTM) predictors, benchmarked at 5/10/20/30 ms, each with an XAI output, plus write-up.

**Data**
- [x] `generate_trajectories.py` written
- [ ] Actually run it; sanity-check the 200 trajectories (plots, value ranges)

**EKF**
- [x] `__init__`, `predict`, `update`, `predict_future` written
- [ ] `test_ekf.py` — tracks a CLEAN trajectory (correctness test)  ← YOU ARE HERE
- [ ] `test_ekf.py` — tracks a NOISY trajectory (usefulness test)
- [ ] Upgrade `predict` uncertainty from `P += Q` to `F @ P @ F.T + Q`
- [ ] Latency benchmark
- [ ] XAI: innovation sequences

**Physics-ODE**
- [ ] `ode_predictor.py` written
- [ ] Tested against synthetic data
- [ ] Latency benchmark
- [ ] XAI: parameter attribution (gravity / drag / v0)

**LSTM**  ← candidate to CUT (ask Prof. Sowah; biggest time sink)
- [ ] `lstm_predictor.py` — architecture + dataset prep + training loop
- [ ] Trained (hours on CPU; run overnight)
- [ ] Latency benchmark
- [ ] XAI: gradient-based importance

**Wrap-up**
- [ ] Benchmark all predictors under 5/10/20/30 ms budgets
- [ ] Comparison analysis (error vs latency)
- [ ] `docs/O2_Findings_Interim.md` (mirror O1 structure)
- [ ] Commit to GitHub

---

## O3 — Latency–accuracy–interpretability tradeoff  (0%)
Cannot start until O2 results exist. Mostly analysis of O1+O2 data.
- [ ] Combine O1 + O2 result tables
- [ ] Build tradeoff curves (accuracy vs inference budget)
- [ ] Identify feasible operating points
- [ ] Headline tradeoff figure
- [ ] `docs/O3_Findings_Interim.md`

---

## O4 — Closed-loop prototype  (0%)  ← LONG POLE, START HARDWARE NOW
- [ ] **Order / confirm hardware** (edge device, camera, servo/actuator, Arduino, mounts) — DO THIS FIRST, blocks everything else in O4
- [ ] Dataset polygon fix in Roboflow (~1 hr, do any time)
- [ ] Assemble rig + flash edge device + install stack (ROS 2, DepthAI SDK)
- [ ] Integrate pipeline: detection → tracking → prediction → actuation
- [ ] 108-trial evaluation (4 angles × 3 speeds × 9 repeats)
- [ ] `docs/O4_Findings_Interim.md`

---

## Thesis (runs alongside; don't leave to the end)
- [x] O1 written up
- [ ] O2 chapter
- [ ] O3 chapter
- [ ] O4 chapter
- [ ] Intro / lit review / methodology / conclusion consolidation
- [ ] Full revision pass

---

## Weekly self-check questions
1. Did O4 hardware move this week? (If no for two weeks running, the 6-week target is gone — escalate.)
2. Did the thesis get real hours, or only the code?
3. What's the ONE current step, and is it written on the CURRENT STEP line above?
