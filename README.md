## Evaluation protocol

*This section was written and committed before any model was trained. It fixes the metrics, the
operating point and the cost assumptions in advance, so that none of them can be selected after
seeing a result.*

### Use case

The model is written for the credit control team of a **wholesale supplier** that sells on trade
credit — goods are shipped first and paid for 60 days later, so every active counterparty carries
an unsecured receivable. The assumed portfolio is 2000 counterparties.

Once a month the model scores the entire portfolio and ranks it by bankruptcy risk. The **top 3%
(~60 companies)** go into a queue for manual review by an analyst, who has access to information
the model does not have: payment history with us, credit bureau records of arrears towards other
creditors, court and insolvency filings, and news. A full review takes roughly two hours; three
analysts spending about 35% of a 120-hour working month on portfolio monitoring have 126 hours
between them, or 63 reviews. That is rounded down to 60, so the operating point falls on a round
3% of the portfolio.

The model does not decide anything. It does not replace the review, and it is not expected to
predict bankruptcy more accurately than an analyst with bureau data. Its only job is to order the
queue: someone has to be looked at first, and the ranking should beat looking at counterparties in
arbitrary order. A flag has no consequence for the counterparty and is invisible to them, which is
what makes the cost of a false alarm small (see below).

In production a second, higher threshold is usually layered on top for automatic action, such as
reducing a credit limit. That is deliberately out of scope here: 4387 training rows and 306
positives do not justify automated decisions with consequences for third parties, and an automatic
action would also destroy the labels — a counterparty whose terms were cut can no longer be
observed defaulting under the original terms.

### Primary metric — PR-AUC

**Average precision (PR-AUC)** is the primary metric. Every result is reported as a lift over the
random-scoring baseline, which equals the positive rate: **0.070**.

ROC-AUC is not used as a headline number. With 7% positives, the false positive rate is computed
against a large negative class, so a substantial number of false alarms moves it very little; the
precision-recall curve is measured against the positives and stays sensitive to exactly the errors
the user pays for. ROC-AUC does appear in the feature analysis, but only as a per-feature ranking
device, never as a model score.

Accuracy is not reported at all. A model that labels every company as solvent reaches 93% accuracy
and catches nothing — a metric that a constant answer wins does not measure anything here.

### Secondary metric — precision@top-3%

The business question is not "how good is the ranking overall" but "of the companies the team
actually has time to review, how many were worth reviewing". That is **precision@top-k**, with k
set by review capacity rather than by a round number.

**k is fixed as a share of the scored population, not as a count.** A cross-validation fold holds
about 878 rows while the holdout holds 1463. A fixed k = 60 would mean the top 6.8% of a fold against
the top 4.1% of the holdout — two different points on the precision-recall curve, where precision
falls as you go deeper into the list. The two numbers would differ for arithmetic reasons before
the model contributed anything. Expressed as a share, the operating point is the same everywhere:
26 companies per fold, 43 in the holdout.

The cost of this choice is granularity. With 26 companies in a fold, precision moves in steps of
1/26 ≈ 0.04 and the spread across folds is wide. This is reported, not smoothed over.

### Cost of errors

Both error types are expressed in money, under the assumptions listed below.

| | Event | Cost |
|---|---|---|
| **FN** | A bankruptcy is missed. The receivable is written off; as an unsecured creditor the supplier recovers ~15% | 40 000 × 0.85 = **34 000 PLN** |
| **FP** | A solvent company is reviewed for nothing. Two analyst-hours are spent; the counterparty never learns of it | ≈ **200 PLN** |

**C = cost(FN) / cost(FP) ≈ 170.**

The conclusion drawn from this is not a threshold but the absence of one as a binding constraint.
At a ratio of this order, a review pays for itself if it catches a bankruptcy roughly once in 170
attempts — far below any precision the model needs to reach. It is therefore worth reviewing every
company the team can physically process, and the queue length is set by 126 analyst-hours per
month, not by the economics of the error.

This conclusion is insensitive to the exposure figure. Across a plausible range of 50 000 to
300 000 PLN of average exposure, C moves between roughly 210 and 1300 — two to three orders of
magnitude in every case, and the reasoning above is unchanged. A cost-based threshold becomes the
binding constraint only if the response to a flag becomes expensive, for example an automatic
reduction of credit terms; under manual review it does not.

### Reporting rules

- Validation is `RepeatedStratifiedKFold` (5 splits × 5 repeats), giving 25 fits per model.
- Every metric is reported as **mean ± std across folds**. A mean without a spread is not
  interpreted.
- **Between two results reported independently, a difference smaller than the sum of the two
  standard deviations is reported as indistinguishable**, not as an improvement. With about 61
  positives in a validation fold, fold-to-fold variance is comparable to the differences between
  models, and a table of results that refuses to name a winner is the honest outcome rather than a
  weak one.
- **Decisions about construction** — whether a column is dropped, whether a transformer earns its
  place — are settled by a paired comparison instead. Both configurations run on the same folds,
  the per-fold differences are taken, and the reported figure is their mean and twice its standard
  error, `2 * std / sqrt(n)`. Sharing the folds cancels the fold-to-fold spread: differences
  scatter by about 0.003 where the metric itself scatters by 0.042, so effects an order of
  magnitude below the threshold above become visible. The standard error assumes independent
  measurements, and 25 folds are 5 splits repeated 5 times with overlapping rows, so the true
  uncertainty is somewhat larger than the formula gives — comfortable for telling 0.08 from 0.000,
  not a test to lean on near the boundary.
- The holdout is evaluated **once**, at the end, with the final pipeline. Every intermediate
  decision is made on cross-validation.
- With ~102 positives in the holdout, the confidence interval on the final figure is wide; a gap
  between the CV estimate and the holdout figure that falls inside it is not interpreted as
  overfitting or as a surprise.

### Assumptions

Both cost figures are invented. The dataset contains financial ratios and a bankruptcy label and
nothing about who might be lending to these companies, so the business context around it has to be
supplied. The numbers were chosen to be internally consistent — 2000 counterparties at 40 000 PLN
of average exposure implies roughly 80 M PLN of receivables and, at 60-day terms, annual revenue
near 480 M PLN, a mid-sized wholesaler where a credit control team of three is plausible.

Exposure is assumed equal across counterparties, which it never is: in a real portfolio a handful
of large accounts carry much of the receivable, the cost of a false negative varies by orders of
magnitude between counterparties, and the queue would be ranked by `score × exposure` rather than
by score alone. The dataset carries no exposure figures, so the simplification is unavoidable. It
is repeated in Limitations.

## Baselines

Fixed before any tuned model, on cross-validation over the training split.

| | PR-AUC | precision@top-3% |
|---|---|---|
| Random ranking | 0.070 | 0.070 |
| Altman Z' (1983) | 0.279 ± 0.049 | 0.442 ± 0.099 |
| Logistic regression | 0.348 ± 0.039 | 0.492 ± 0.081 |

The regression is deliberately plain: median imputation, standard scaling,
`class_weight="balanced"`, all 64 features, no selection.

The gap of 0.069 sits below the combined spread of 0.088 and is not read as an improvement — even
though Altman's weights were calibrated on US manufacturing firms in the 1970s and saw none of this
data. Everything that follows is measured against 0.279.

## Feature set

The file is the five-year subset of the Polish companies bankruptcy data (UCI 365): predictions
one year ahead, 64 financial ratios named `Attr1`–`Attr64`. The names are kept as they are so that
every column stays traceable to the UCI description; `FEATURE_LABELS` in `src/config.py` carries
the readable descriptions for plot captions only.

After removing 60 groups of exact duplicates the data is 5850 × 65 with 408 positives (7.0%). The
stratified split leaves 4387 training rows with 306 positives and 1463 holdout rows with 102.

Reaching the model: **63 ratios** (`Attr21` excluded, see below) plus **6 missingness indicators**
(`Attr24`, `Attr27`, `Attr28`, `Attr37`, `Attr41`, `Attr45`). The indicators are not an arbitrary
selection: 49 columns contain missing values, 11 of them in more than 1% of rows, and several
missingness masks coincide bitwise — the inventory block and the fixed-asset block collapse into
one indicator each, with the lowest-numbered column standing for the block. Whether a value is
missing carries signal on its own; the fact that `Attr27` is absent separates the classes at
ROC-AUC 0.627.

No engineered features were added. Two candidates were tested and both were rejected on measured
grounds, recorded in `notebooks/05_feature_engineering.ipynb`.

## Decisions

| Decision | Reason |
|---|---|
| Everything after the split is done on the training rows only | Transformation choices are made after looking at the data, so the holdout has to be unseen when they are made |
| Two pipeline factories rather than one with a branch parameter | The branches share only the indicator block; a single factory would be a switch statement pretending to be a pipeline |
| The model is an argument to the factory | The step is named `model` in one place, which is what `RandomizedSearchCV` addresses |
| `feature_cols` is a parameter, `MISSING_INDICATOR_COLS` comes from config | The feature list moves between experiments; the indicator list was fixed by the missingness analysis |
| `MissingIndicator(features="all")` rather than the default | The default decides which columns to emit from the data, so the output width would float between folds |
| `remainder="drop"` written out explicitly | Every column is named in a triplet, but the decision about the remainder should be visible — `Attr21` is what lands there |
| `set_output(transform="pandas")` on both branches | `feature_names_in_` then catches column mismatches, and names are needed for SHAP and for reading `coef_` |
| `Attr21` dropped entirely | The whole effect sits in 80 rows where the value is missing, 78 of them bankrupt; on the other 4307 rows the column is worth 0.0054 ± 0.0111. Measured price: 0.734 instead of 0.817 PR-AUC in the boosting branch |
| Missingness indicators kept in the boosting branch despite measuring −0.0004 ± 0.0006 | Measured at default hyperparameters; NaN routing changes after tuning, so the result is worth re-checking rather than acting on |
| `QuantileTransformer` inside the linear pipeline | Tails reach 694× the 99th percentile; `StandardScaler` is linear and does not change shape |
| `n_quantiles=500` | About 3510 rows in the training part of a fold; the default of 1000 would copy the sample |
| Median imputation rather than a constant | The mean is inflated by the tails, and a constant becomes a subgroup marker for a tree |
| Feature importance is read by blocks | At a correlation of 0.999 importance is split arbitrarily within a group |
| `RepeatedStratifiedKFold`, 5 × 5 | 7% positives, about 61 of them per validation fold |
| The holdout is evaluated once, at the end | Anything else is fitting to the test set |
| `cross_validate_model` and `compare_pipelines` return per-fold arrays | A mean can always be derived from the folds; the folds cannot be recovered from a mean |
| Ties in precision@top-k are broken by row order | It matches how the queue would actually be filled |
| Altman is wrapped in a scikit-learn compatible class | One interface, identical folds, a fair comparison |
| No negative-equity flag | −0.0015 ± 0.0022 PR-AUC. Risk varies with the level of equity rather than with its sign, and the level is already in `Attr10`; binarising it at the accounting boundary only coarsens what the model has |
| No correlation-based feature selection | −0.0097 ± 0.0095 at a 0.99 cut and −0.0285 ± 0.0135 at 0.95, the loss growing with the cut. With 63 features against 4387 rows there is no penalty for keeping duplicated columns, and L2 already stabilises the weights inside a correlated group |

## Limitations

**Selection bias from pre-filtering.** The ARFF header records
`SubsetByExpression -E not ismissing(ATT20)`: companies with no current-year revenue were removed
before the file was published. The population the model is fitted on is therefore not the
population it would score in production, and the direction of the bias is unknown.

**`Attr21` is probably a collection artefact, and its price is known.** Missingness in the column
predicts the target almost perfectly — 78 of the 80 rows with a missing value are bankrupt — and
tree models exploit it whether or not an explicit indicator is supplied, since the marker can be
reconstructed from the column itself. Constant imputation does not help either: the constant
becomes the subgroup label. A company that is still trading has last year's filing by definition,
so the pattern should not recur in production. This cannot be settled inside the data, because the
same contamination is present in the holdout in the same proportion, and every internal estimate
would agree with itself while being wrong. The column was dropped and the cost measured: 0.734
against 0.817.

**No feature selection was performed, and that conclusion does not travel.** Correlation-based
selection was measured and rejected, but the measurement holds for 63 features, 4387 rows and a
regularised linear model. With many more features, fewer rows, or an unregularised model the trade
would look different.

**Near-duplicate detection is conservative.** Near-duplicates were searched by rounding all
features to k decimals for k = 1..6 and then looking for exact matches. Five pairs turned up, none
containing a positive, and the estimated effect on PR-AUC is zero, so validation stays row-wise and
`GroupKFold` is not needed. The method finds rows that are close on every feature at once; it would
miss rows that differ substantially on one ratio.

**The split is random, not temporal.** All companies in the file are observed over the same
period, so there is no way to train on earlier years and validate on later ones. A model that is
put into production is asked to generalise forward in time, and a random split does not measure
that. Macroeconomic conditions shift, and with them the base rate.

**Every business assumption is invented.** Exposure, the cost of each error type, the portfolio
size and the capacity of the credit control team are not in the dataset and were supplied to give
the metrics an operating point. They are internally consistent, not observed.

**Exposure is treated as equal across counterparties.** In a real portfolio a few large accounts
carry much of the receivable, so the cost of a false negative varies by orders of magnitude and the
queue would be ordered by `score × exposure`. The dataset carries no amounts, so the simplification
is unavoidable.
