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
analysts spending about a third of their time on portfolio monitoring absorb about 60 reviews a
month.

The model does not decide anything. It does not replace the review, and it is not expected to
predict bankruptcy more accurately than an analyst with bureau data. Its only job is to order the
queue: someone has to be looked at first, and the ranking should beat looking at counterparties in
arbitrary order. A flag has no consequence for the counterparty and is invisible to them, which is
what makes the cost of a false alarm small (see below).

In production a second, higher threshold is usually layered on top for automatic action, such as
reducing a credit limit. That is deliberately out of scope here: 4388 training rows and 306
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
878 rows while the holdout holds 1462. A fixed k = 60 would mean the top 6.8% of a fold against
the top 4.1% of the holdout — two different points on the precision-recall curve, where precision
falls as you go deeper into the list. The two numbers would differ for arithmetic reasons before
the model contributed anything. Expressed as a share, the operating point is the same everywhere:
26 companies per fold, 44 in the holdout.

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
- **A difference smaller than the sum of the two standard deviations is reported as
  indistinguishable**, not as an improvement. With about 61 positives in a validation fold, fold-to
  -fold variance is comparable to the differences between models, and a table of results that
  refuses to name a winner is the honest outcome rather than a weak one.
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
