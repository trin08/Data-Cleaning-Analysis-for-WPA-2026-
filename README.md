# Data-Cleaning-Analysis-for-WPA-2026-
# Human or AI: Accuracy, Confidence and Perceptions amongst Students and Faculty

Analysis pipeline for a survey study on how accurately students and faculty
at Golden West College distinguish AI-generated from human-generated text
and images, and their attitudes toward AI use in academic work.

**Authors:** Tri Nguyen, Kristen Tran, Amy Jennings, Ed.D.
**Institution:** Golden West College, Huntington Beach, CA
**Poster:** [`docs/poster.pdf`]([docs/poster.pdf](https://docs.google.com/presentation/d/1ZabgET3PYoMdP_lYIQTYg7lbckrEAw4UEtc_luUevPM/edit?usp=sharing))

Data analysis code (`clean_data.py`, `score_and_analyze.py`) was written by
Tri Nguyen.

## Study

**Participants:** 83 respondents (68 students, 15 faculty) at Golden West
College.

**Hypotheses:**
1. Students would identify AI-generated content more accurately than
   instructors.
2. Younger individuals and those with higher screen time or social media
   use would identify AI-generated content more accurately.

**Key results:**
- Overall accuracy identifying AI-generated content was low (*M* = 6.59,
  *SD* = 2.05) and did not differ significantly between students and
  faculty, *t*(81) = 1.56, *p* = .112 — H1 was not supported.
- Confidence in identifying AI-generated text and images dropped
  significantly from pre- to post-assessment (text: *W* = 899.5, *p* =
  .001; images: *W* = 1771.5, *p* < .001), suggesting participants
  overestimated their detection ability going in.
- Social media use showed a weak negative correlation with accuracy,
  *r*(80) = -.28, *p* = .012. Age and screen time were not significantly
  correlated with accuracy — H2 was largely not supported.
- Students rated AI use as more acceptable than faculty did.

**Limitations:** convenience sample from a single community college; small
faculty subsample (*n* = 15) relative to students (*n* = 68); self-reported
screen time, social media use, and confidence measures.

Full method, survey instruments, and discussion are in the poster.

## Pipeline

1. **`clean_data.py`** — takes the raw Google Forms export and produces
   `Cleaned_WPA_Dataset.csv`.
2. **`score_and_analyze.py`** — consumes the cleaned CSV and produces every
   statistic reported on the poster, plus supporting exploratory analyses.

```
python clean_data.py
python score_and_analyze.py
```

## Repository structure

```
wpa-human-or-ai/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore              # excludes data/raw and data/processed
├── src/
│   ├── clean_data.py
│   └── score_and_analyze.py
├── data/
│   ├── raw/                 # not tracked — place the Google Forms export here
│   └── processed/           # not tracked — Cleaned_WPA_Dataset.csv lands here
└── docs/
    └── poster.pdf
```

Survey data is not committed to this repository. To reproduce the results,
place the raw Google Forms export in `data/raw/` and run the two scripts
in order.

## Design notes

**Why the pipeline is split into two stages.** Cleaning and analysis were
kept separate so the cleaned dataset can be inspected or reused on its own,
and so a change to scoring logic doesn't require re-running the (slower,
more failure-prone) raw-data cleaning step.

**Column selection by name, not position.** Item groups (e.g. the 10
student-acceptability items) are selected by matching text that appears in
the question itself, rather than by column index. Google Forms exports can
reorder columns if the form is edited, and index-based selection fails
silently in that case.

**Missing data is not imputed.** Missing Likert-scale responses are left as
`NaN`; every downstream mean, correlation, and t-test uses pairwise deletion
(`nan_policy='omit'` / `.dropna()`) rather than imputing a value the
respondent never gave.

**Likert parsing handles multi-select artifacts.** A few respondents
multi-selected on what should have been single-select scale items (e.g. a
raw value of `"4, 5"`). `parse_likert` finds every digit token in the cell
and averages them, so `"4, 5"` becomes `4.5` rather than silently keeping
just the first number.

**Image items are keyed by position, text items by name.** The seven image
items share an identical question stem ("Do you believe this image is...")
and can only be distinguished by their on-screen order — this is a
constraint of the source form, so `IMAGE_ANSWER_KEY` must be kept in sync
if the form's image order ever changes. Text items are self-labeled
(1.A, 1.B, 2.A, ...) and are matched by name instead.

**Role/field/platform classification is heuristic.** `classify_role`,
`classify_field`, and `classify_platform` use keyword matching on
free-text/checkbox responses. These are coarse proxies, not validated
instruments, and are treated as exploratory rather than confirmatory.

## Known limitations

- Field-of-study and platform-type classification rely on keyword lists
  that may not generalize beyond this sample's phrasing.
- Image ground truth is order-dependent (see above) rather than
  self-describing in the data.

## AI assistance disclosure

The data analysis code was written by Tri Nguyen. Portions of this
pipeline (refactoring, docstrings, and the compound-number age-parsing fix)
were developed with AI assistance.
