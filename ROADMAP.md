# Poniard roadmap

**Thesis:** Poniard is a lightweight scikit-learn companion for *multi-model diagnostics*.
Compare models to get oriented, then answer *where they fail, whether differences are real,
and what to try next* — then export a plain sklearn object and leave.

Not AutoML. Not end-to-end. Every feature must earn its place.

**Persona priority**

1. **Primary (B):** where do models fail / what should I fix in the data or features?
2. **On-ramp (A):** which model family is even viable on this task?
3. **Exit (C):** what do I export and keep working on outside Poniard?

**Constraints**

- Stay lightweight: core deps remain sklearn/pandas/numpy/scipy/joblib/tqdm; plotly optional.
- No XGBoost/LightGBM/CatBoost (or similar) as dependencies — user installs and injects.
- Pure numpy/scipy for stats (no new stats deps).
- Library is effectively unused → **breaking changes are fine**. Prefer delete over deprecate.
- Features need a clear "I'd install for this" pull. Commodity wrappers get cut or starved.

**Investment rank**

1. Error analysis (product core)
2. Prediction similarity → ensemble (closed loop)
3. Statistical comparison ("is A really better?")
4. Time/quality surface
5. Tune-as-experiment-glue (sharpen or cut)
6. Plotting (support layer for 1–4)
7. Core compare + preprocessor (maintain, don't expand)

---

## 0. Reposition and shrink surface

**Goal:** Make the wedge obvious in README and API. Kill noise.

### 0.1 README rewrite
- Lead with diagnostics, not "fit many models."
- On-ramp: compare → results.
- Core path: error analysis → similarity/ensemble → export.
- Plotting mentioned as optional notebook sugar, not a pillar.
- Drop feature laundry lists that sound like PyCaret-lite.

### 0.2 Public API diet (break freely)

**Done**
- [x] Delete public `decision_function` (CV path remains via `_predict` for plots)
- [x] Drop `TargetEncoder` re-export from `poniard.preprocessing`
- [x] Drop `DateLevel` from package `__all__` (still on `datetime` module for power users)
- [x] Plots: **left alone** (optional satellite; revisit later)
- [x] `setup` kept (audit mutator workflow)
- [x] `tune_estimator` redesigned as experiment glue (see §5 done items)
- [x] Full README wedge rewrite (with P1 error-analysis story)
- [x] `compare()` added (paired fold comparison)
- [x] `pareto()` and `best_under()` added (time-quality surface)

**Still open**
- Plot diet (deferred — owner review)
- Diversity ensemble (P2)

**Ship when:** README tells the new story in one screen; public surface is small enough to hold in your head.

---

## 1. Error analysis as the product

**Goal:** The reason someone installs Poniard. Screenshot-worthy multi-model failure forensics.

**Status: Done** — `ErrorReport` dataclass with universal failures, disagreement set, lift by target/feature.

### 1.1 Report shape (wow minimum)
`analyze()` (or a dedicated report object) should make these trivial:

- [x] **Universal failures:** samples every selected model gets wrong (`freq == n_estimators`).
- [x] **Disagreement set:** samples where models split (useful for ensembling and labeling).
- [x] **Lift vs baseline:** per target class/bin and per feature value — error rate vs global error rate (not just raw counts).
- [ ] **Top slices:** feature values / bins where error lift ≥ threshold (e.g. 2×), ranked.
- [x] **Per-estimator summary:** n_errors, error_rate, mean confidence-of-wrong / residual.
- [ ] **Stable indices:** sample ids that survive CV prediction alignment (document assumptions hard).

### 1.2 API cleanup
- [x] First-class report type (`ErrorReport` dataclass) instead of a loose dict.
- [x] `from_poniard` default: analyze all non-dummy fitted estimators if names omitted.
- [ ] Avoid recompute traps: reuse cached `cross_val_predict` / proba from the Poniard session when present.
- [x] Clear separation: ranking definition (classif vs reg) stays explicit and documented.

### 1.3 Stretch (after minimum wow)
- [ ] Simple cohort labels ("high cardinality category X", "target bin top decile").
- [ ] Optional short text summary for notebooks (`report.narrative()` — careful, no LLM deps; templated stats only).
- [ ] Hook points for plots: error lift bars, universal-failure table, disagreement heatmap.

**Ship when:** A user can run `ErrorAnalyzer.from_poniard(clf).analyze(X, y)` and immediately answer:
"what rows are toxic to every model?", "which slices are 3× worse?", "where do models disagree?"

**Non-goals:** full causal inference, SHAP stack, AutoML root-cause novels.

---

## 2. Diversity → ensemble closed loop

**Goal:** Similarity stops being a trivia method and starts driving decisions.

### 2.1 `build_ensemble` default becomes diversity-aware
When results + predictions exist:

1. Rank by primary metric (exclude dummies).
2. Greedily pick members that stay strong **and** low pairwise similarity on errors (or predictions).
3. Fall back to pure `top_n` if similarity can't be computed.

API sketch:

```python
clf.build_ensemble(
    method="stacking",  # or "voting"
    strategy="diversity",  # default; also "top_n"
    top_n=3,
    # optional knobs: min_metric, similarity_threshold, sort_by
)
```

### 2.2 Wire the data
- `get_predictions_similarity` remains public and is what ensemble uses under the hood.
- Cache CV predictions used for similarity so ensemble doesn't silently recompute forever.
- Document: similarity `on_errors=True` default and when to flip it.

### 2.3 After ensemble
- Ensemble is just another named pipeline → `fit` adds it to the comparison table (already true).
- Error analysis can include the ensemble and show whether universal failures shrank.

**Ship when:** Default ensemble picks a different (and defensible) set than naive top-3 on a dataset where two trees fail alike and a linear model fails differently.

---

## 3. Statistical comparison

**Goal:** Stop pretending fold-mean leaderboards are truth.

**Status: Done** — `compare()` method on `PoniardBaseEstimator`.

### 3.1 Paired fold comparison (pure numpy/scipy)
- [x] Operate on per-fold scores already in `_experiment_results`.
- [x] Pairwise tests appropriate for CV folds (document limitations honestly — folds aren't independent).
- [x] Practical outputs people use:
  - [x] mean diff + CI
  - [x] win/tie/loss across folds
  - [ ] simple ranking that resists noise (e.g. mean rank, or CD-diagram data)

### 3.2 API sketch

```python
clf.compare()                    # all models, primary metric
clf.compare(metrics=["f1", "roc_auc"])
clf.compare(estimators=["LogisticRegression", "RandomForestClassifier"])
```

Returns a small results object / DataFrames: pairwise table + optional ranking summary.

### 3.3 Honesty in docs
- [x] State clearly: this is **exploratory comparison**, not a paper-grade multiple-testing shrine.
- [x] Prefer methods implementable without new deps (paired t on fold scores, Wilcoxon, bootstrap CI — pick one solid default + escape hatches).

**Ship when:** User can answer "is RF actually better than LR on this CV, or are we reading noise?" without leaving Poniard.

**Non-goals:** full bayesian hierarchical CV models, author-grade critical difference plot dependency stacks (plot later if easy in plotly).

---

## 4. Time / quality surface

**Goal:** Practical model choice, not only peak metric.

**Status: Done** — `pareto()` and `best_under()` methods.

### 4.1 Productize what you already measure
- [x] `fit_time` / `score_time` already exist in results.
- [x] Add a first-class view: metric vs log(fit_time) table or Pareto filter.
- [x] Helpers like "best under T seconds (mean fit_time)" and "within r% of best metric, pick fastest."

### 4.2 API sketch

```python
clf.get_results()                  # stays
clf.pareto(metric=None)            # metric vs time, non-dominated
clf.best_under(seconds=2.0)        # name or row
```

### 4.3 Plot support (optional)
- [ ] Single scatter: time vs metric, dummy annotated — only if cheap.

**Ship when:** Choosing "good enough and cheap" is one call, not manual dataframe wrangling.

---

## 5. Tune-as-experiment-glue

**Goal:** Better than `get_estimator` → tune outside → `add_estimators` because the
tuned model re-enters the same experiment with zero prep/CV drift.

### 5.1 Done
- [x] No default grids
- [x] Bare param names auto-prefixed (`{"C": [...]}` → `{name}__C`); keys with `__` untouched
- [x] Side-by-side re-entry default `{name}_tuned`; refuse silent overwrite; custom name OK
- [x] Unknown baseline / empty grid / bad mode → clear errors
- [x] `get_tuning_results(name?)` → `best_params_`, `best_score_`, resolved `grid`, fitted `search`
- [x] README documents the experiment-loop story

### 5.2 Still open
- [ ] Paired baseline-vs-tuned fold delta once `compare()` (§3) exists
- [ ] Optionally surface tune summary in `get_results` metadata (low priority)

**Ship when:** §3 can answer “did tuning actually help on these folds?”

---

## 6. Plotting (optional satellite)

**Goal:** Notebook sugar that serves the wedge. Not a product pillar.

### 6.1 Keep / add
- Metrics overview (on-ramp)
- Overfitness
- Confusion / ROC / residuals (task-appropriate basics)
- **New, high value:** error lift bars, disagreement / similarity heatmap, time-vs-quality scatter
- `full_estimator_analysis` only if it stays thin and routes through the same data as reports

### 6.2 Cut or bury
- Partial dependence unless tied to error slices
- Anything that duplicates sklearn/plotly one-liners without diagnostic context

### 6.3 Messaging
README: "optional visual analysis" — one short section, not a hero feature.

**Ship when:** plots mostly visualize §1–4 outputs; no orphan chart APIs.

---

## 7. Core compare + preprocessor (maintenance mode)

**Goal:** Reliable on-ramp. No feature sprawl.

### 7.1 Keep sharp
- Type inference + `reassign_types`
- Default prep that is boring and correct
- Dummy baseline always present
- Clean `get_estimator` exit (no poniard references)
- Fitted tracking, save/load, side-effect discipline (already invested — don't regress)

### 7.2 UX nits worth breaking for
- Prefer one obvious happy path (`fit` configures if needed; `setup` remains for mutators between configure and fit, or collapse if setup is pure friction).
- Naming rules stay boring and documented.

### 7.3 Explicit non-goals
- Batteries-included GBDT stacks
- Default search spaces
- Deep AutoML / pipeline search
- Target leakage "magic" feature engineering

---

## Suggested delivery order

Break into small releases; each should be install-worthy alone.

| Phase | Theme | Status |
|---|---|---|
| **P0** | Reposition + API diet + README | Done |
| **P1** | Error analysis report wow (universal fails, lift, disagreement) | Done |
| **P2** | Diversity-default ensemble + prediction cache | Done |
| **P3** | Statistical `compare()` | Done |
| **P4** | Pareto / best_under time-quality | Done |
| **P5** | Tune glue redesign **or** deletion | Done |
| **P6** | Plot pass aligned to P1–P4 | Done |

Parallelism: P0 first. P1 can start immediately after. P2 depends on similarity + cached preds (partially exists). P3 reads fold scores (exists). P4 is small and can slip between larger phases. P5 last among core so compare/error can support tuned deltas. P6 continuously skims off P1–P4.

---

## Definition of done (project-level)

Poniard is "done enough" for this arc when a new user can:

1. Fit a handful of models and see a sane leaderboard with dummy baseline.
2. Run error analysis and point to toxic rows + high-lift slices + disagreements.
3. Build a diversity-aware ensemble and see it appear in the same table.
4. Ask whether model A beats B beyond fold noise.
5. Pick a fast-enough model, export a pure sklearn pipeline, uninstall Poniard mentally.

If a proposed feature doesn't make one of those five sharper, it doesn't ship.

---

## Open implementation notes (resolve during P0/P1)

- Report type: dataclass vs thin class with `__repr__` tables.
- Default similarity metric for ensemble selection (error agreement vs prediction association).
- Which CV-aware statistical default (pick one; document caveats).
- Whether `setup` remains public or becomes an advanced escape hatch.
- How aggressively to cache `cross_val_predict` outputs (memory vs UX).
