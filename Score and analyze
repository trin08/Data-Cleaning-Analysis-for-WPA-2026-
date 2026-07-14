"""
score_and_analyze.py
=====================
Stage 2 of the WPA "Human or AI?" analysis pipeline.

Consumes Cleaned_WPA_Dataset.csv (output of clean_data.py) and produces
every statistic reported on the WPA poster, plus the exploratory analyses
that motivated it.

Column groups are selected by matching text in the question itself
(e.g. "if used by STUDENTS", "Fear of Losing Research Skills") rather
than by column position, so the script stays correct if columns are
reordered upstream.

Run:
    python score_and_analyze.py
"""

import re

import numpy as np
import pandas as pd
import scipy.stats as stats

INPUT_FILENAME = "Cleaned_WPA_Dataset.csv"

# ---------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------

df = pd.read_csv(INPUT_FILENAME)


# ---------------------------------------------------------------------
# 2. COLUMN GROUPS
# ---------------------------------------------------------------------

COL_STUDENT_ACCEPT = [c for c in df.columns if "if used by STUDENTS" in c]
COL_FACULTY_ACCEPT = [c for c in df.columns if "if used by FACULTY" in c]
COL_CONCERN_ACCURACY = [c for c in df.columns if c.startswith("Concern About the Accuracy")]
COL_PLAGIARISM = [c for c in df.columns if c.startswith("Fear of Committing Plagiarism")]
COL_GUIDELINES = [c for c in df.columns if c.startswith("Lack of Institutional Guidelines")]
COL_SKILL_LOSS = [c for c in df.columns if c.startswith("Fear of Losing Research Skills")]
COL_IMAGE_ITEMS = [c for c in df.columns if c.startswith("Do you believe this image is")]
COL_TEXT_ITEMS = [
    c for c in df.columns
    if c.startswith(("1.A)", "1.B)", "2.A)", "2.B)", "3.A)", "3.B)", "3.C)"))
]

# Single item (of the 5-item Guidelines composite) called out separately
# as the clearest evidence of a policy vacuum.
COL_NO_GUIDANCE_ITEM = next(
    c for c in COL_GUIDELINES if "no clear guidance from universities" in c
)

# Single item (of the 4-item Plagiarism composite) called out separately
# as the most direct integrity-risk item.
COL_ETHICS_VIOLATION_ITEM = next(
    c for c in COL_PLAGIARISM if "violates academic integrity" in c
)

COL_PRE_FEELINGS = next(c for c in df.columns if c.startswith("Before doing a measure"))
COL_POST_FEELINGS = next(c for c in df.columns if c.startswith("After doing a test"))
COL_PRE_CONF_TEXT = "How confident are you in identifying AI-generated text, such as assignments or research papers?"
COL_POST_CONF_TEXT = COL_PRE_CONF_TEXT + ".1"
COL_PRE_CONF_IMAGE = "How confident are you in identifying AI-generated images on social media or other online platforms?"
COL_POST_CONF_IMAGE = COL_PRE_CONF_IMAGE + ".1"

COL_ENCOUNTER_PLATFORM = "Where do you usually encounter AI-generated content?  (Choose all that apply) "

# Ground-truth answer keys, in on-screen survey order.
# Image items share an identical question stem ("Do you believe this
# image is") and can only be told apart by on-screen position; if the
# form's image order changes, this key must be updated to match.
IMAGE_ANSWER_KEY = ["Real", "Real", "AI", "Real", "AI", "AI", "AI"]

# Text items are self-labeled (1.A, 1.B, 2.A, ...) so they're matched by
# name rather than position.
TEXT_ANSWER_KEY = {
    "1.A": "Real", "1.B": "AI",
    "2.A": "Real", "2.B": "AI",
    "3.A": "Real", "3.B": "AI", "3.C": "AI",
}

# Keyword heuristics for coarse role/field/platform classification.
# These are lightweight text-matching proxies, not validated instruments.
FACULTY_KEYWORDS = ("faculty", "professor", "instructor")
STEM_KEYWORDS = ("stem", "science", "tech", "engineer", "math", "bio", "chem", "phys", "nurs", "health")
HUMANITIES_KEYWORDS = ("art", "english", "hist", "soc", "psych", "bus", "law", "educ", "human", "communication")
VISUAL_PLATFORM_KEYWORDS = ("tiktok", "instagram", "youtube")
TEXT_PLATFORM_KEYWORDS = ("x/twitter", "facebook")


# ---------------------------------------------------------------------
# 3. LIKERT PARSING
# ---------------------------------------------------------------------

def parse_likert(value):
    """Extract a 1-5 Likert rating from a raw survey cell.

    Handles plain numbers ("4"), scale-labeled numbers ("5: Strongly
    Agree"), and multi-select artifacts ("4, 5: Strongly Agree") by
    averaging every digit token found. Returns NaN for missing or
    unparseable cells.
    """
    if pd.isna(value):
        return np.nan
    numbers = re.findall(r"\d+", str(value))
    if not numbers:
        return np.nan
    return np.mean([float(n) for n in numbers])


def parse_likert_columns(columns):
    for col in columns:
        df[col] = df[col].apply(parse_likert)


parse_likert_columns(
    COL_STUDENT_ACCEPT
    + COL_FACULTY_ACCEPT
    + COL_CONCERN_ACCURACY
    + COL_PLAGIARISM
    + COL_GUIDELINES
    + COL_SKILL_LOSS
    + [COL_PRE_CONF_TEXT, COL_POST_CONF_TEXT, COL_PRE_CONF_IMAGE, COL_POST_CONF_IMAGE]
)

df["Social_Media_Time_Daily"] = pd.to_numeric(df["Social_Media_Time_Daily"], errors="coerce")
df["Screen_Time_Daily"] = pd.to_numeric(df["Screen_Time_Daily"], errors="coerce")


# ---------------------------------------------------------------------
# 4. DERIVED FIELDS
# ---------------------------------------------------------------------

def classify_role(value: str) -> str:
    s = str(value).lower()
    if "student" in s:
        return "Student"
    if any(x in s for x in FACULTY_KEYWORDS):
        return "Faculty"
    return "Other"


def classify_field(value: str) -> str:
    s = str(value).lower()
    if any(x in s for x in STEM_KEYWORDS):
        return "STEM"
    if any(x in s for x in HUMANITIES_KEYWORDS):
        return "Humanities/Social"
    return "Other"


def classify_platform(value: str) -> str:
    """Rough visual-vs-text-platform heuristic based on selected platforms.
    Coarse proxy (multi-select checkbox text), not a validated media-diet
    measure -- treat results as exploratory."""
    s = str(value).lower()
    visual = any(x in s for x in VISUAL_PLATFORM_KEYWORDS)
    text = any(x in s for x in TEXT_PLATFORM_KEYWORDS)
    if visual and not text:
        return "Visual"
    if text and not visual:
        return "Text"
    return "Mixed"


df["Role_Clean"] = df["Role"].apply(classify_role)
df["Field_Group"] = df["Field_of_Study"].apply(classify_field)
df["Platform_Type"] = df[COL_ENCOUNTER_PLATFORM].apply(classify_platform)


def score_item(answer, correct: str) -> float:
    """1 if the respondent's Real/AI judgment matches ground truth, else 0.
    Returns NaN (not 0) for a blank response so accuracy is computed only
    over items the respondent actually answered."""
    if pd.isna(answer):
        return np.nan
    a = str(answer).lower()
    guessed_ai = "ai" in a
    guessed_real = "real" in a or "human" in a
    if not guessed_ai and not guessed_real:
        return np.nan
    guess = "ai" if guessed_ai else "real"
    return float(guess == correct.lower())


def false_positive(answer, correct: str) -> float:
    """1 if a genuinely REAL item was misidentified as AI-generated."""
    if correct.lower() != "real" or pd.isna(answer):
        return np.nan
    return float("ai" in str(answer).lower())


# --- Image scoring (position-keyed, see IMAGE_ANSWER_KEY note above) ---
image_scores = pd.DataFrame(
    {
        col: df[col].apply(lambda a, k=key: score_item(a, k))
        for col, key in zip(COL_IMAGE_ITEMS, IMAGE_ANSWER_KEY)
    }
)
image_fp = pd.DataFrame(
    {
        col: df[col].apply(lambda a, k=key: false_positive(a, k))
        for col, key in zip(COL_IMAGE_ITEMS, IMAGE_ANSWER_KEY)
        if key == "Real"
    }
)

# --- Text scoring (name-keyed) ---
def text_key_for(col: str) -> str:
    prefix = col[:4].rstrip(")").strip() + ")"
    for tag, key in TEXT_ANSWER_KEY.items():
        if col.startswith(tag):
            return key
    raise KeyError(f"No answer key found for text column: {col[:40]}")

text_scores = pd.DataFrame(
    {col: df[col].apply(lambda a, c=col: score_item(a, text_key_for(c))) for col in COL_TEXT_ITEMS}
)
text_fp = pd.DataFrame(
    {
        col: df[col].apply(lambda a, c=col: false_positive(a, text_key_for(c)))
        for col in COL_TEXT_ITEMS
        if text_key_for(col) == "Real"
    }
)

df["Image_Accuracy"] = image_scores.mean(axis=1)
df["Text_Accuracy"] = text_scores.mean(axis=1)
df["Total_Accuracy"] = pd.concat([image_scores, text_scores], axis=1).mean(axis=1)
df["False_Positive_Rate"] = pd.concat([image_fp, text_fp], axis=1).mean(axis=1)

# --- Composite scale scores ---
df["Student_Accept_Score"] = df[COL_STUDENT_ACCEPT].mean(axis=1)
df["Faculty_Accept_Score"] = df[COL_FACULTY_ACCEPT].mean(axis=1)
df["Fairness_Gap"] = df["Student_Accept_Score"] - df["Faculty_Accept_Score"]

df["Fear_Plagiarism_Score"] = df[COL_PLAGIARISM].mean(axis=1)
df["Policy_Confusion_Score"] = df[COL_GUIDELINES].mean(axis=1)
df["Fear_Skill_Loss_Score"] = df[COL_SKILL_LOSS].mean(axis=1)
df["Concern_Accuracy_Score"] = df[COL_CONCERN_ACCURACY].mean(axis=1)

# --- Confidence calibration ---
df["Pre_Conf_Norm"] = (df[[COL_PRE_CONF_TEXT, COL_PRE_CONF_IMAGE]].mean(axis=1) - 1) / 4
df["Calibration_Gap"] = df["Pre_Conf_Norm"] - df["Total_Accuracy"]
df["Confidence_Shift"] = (
    df[[COL_POST_CONF_TEXT, COL_POST_CONF_IMAGE]].mean(axis=1)
    - df[[COL_PRE_CONF_TEXT, COL_PRE_CONF_IMAGE]].mean(axis=1)
)


# ---------------------------------------------------------------------
# 5. REPORTING HELPERS
# ---------------------------------------------------------------------

def by_role(col: str):
    return df[df["Role_Clean"] == "Student"][col].dropna(), df[df["Role_Clean"] == "Faculty"][col].dropna()


def sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------
# 6. HEADLINE RESULTS (matches the WPA poster)
# ---------------------------------------------------------------------

print_header("ACCURACY OF IDENTIFYING AI-GENERATED CONTENT")
overall_acc = df["Total_Accuracy"].dropna()
print(f"Overall accuracy: M={overall_acc.mean():.3f}, SD={overall_acc.std():.3f}, n={len(overall_acc)}")

s_acc, f_acc = by_role("Total_Accuracy")
t, p = stats.ttest_ind(s_acc, f_acc, nan_policy="omit")
print(f"Student (n={len(s_acc)}) M={s_acc.mean():.3f} vs Faculty (n={len(f_acc)}) M={f_acc.mean():.3f}")
print(f"Independent t-test: t={t:.2f}, p={p:.3f} {sig_stars(p)}")

print_header("CONFIDENCE: PRE- VS POST-ASSESSMENT (Wilcoxon signed-rank)")
for label, pre_col, post_col in [
    ("Text", COL_PRE_CONF_TEXT, COL_POST_CONF_TEXT),
    ("Image", COL_PRE_CONF_IMAGE, COL_POST_CONF_IMAGE),
]:
    paired = df[[pre_col, post_col]].dropna()
    w, p = stats.wilcoxon(paired[pre_col], paired[post_col])
    print(
        f"{label}: pre Mdn={paired[pre_col].median():.0f} -> "
        f"post Mdn={paired[post_col].median():.0f}, W={w:.1f}, p={p:.4f} {sig_stars(p)}"
    )

print_header("CORRELATIONS WITH ACCURACY")
for label, col in [
    ("Age", "Age"),
    ("Screen time", "Screen_Time_Daily"),
    ("Social media time", "Social_Media_Time_Daily"),
]:
    paired = df[[col, "Total_Accuracy"]].dropna()
    r, p = stats.pearsonr(paired[col], paired["Total_Accuracy"])
    print(f"{label} vs Accuracy: r({len(paired) - 2})={r:.3f}, p={p:.3f} {sig_stars(p)}")


# ---------------------------------------------------------------------
# 7. FAIRNESS / DOUBLE-STANDARD ANALYSIS
# ---------------------------------------------------------------------

print_header("FAIRNESS GAP: 'ACCEPTABLE FOR STUDENTS' VS 'ACCEPTABLE FOR FACULTY'")
for role in ("Student", "Faculty"):
    subset = df[df["Role_Clean"] == role]
    if subset.empty:
        continue
    t, p = stats.ttest_rel(
        subset["Student_Accept_Score"], subset["Faculty_Accept_Score"], nan_policy="omit"
    )
    print(f"\n{role} respondents (n={len(subset)}):")
    print(f"  Rate AI use acceptable for STUDENTS: {subset['Student_Accept_Score'].mean():.2f} / 5")
    print(f"  Rate AI use acceptable for FACULTY:  {subset['Faculty_Accept_Score'].mean():.2f} / 5")
    print(f"  Double-standard gap: {subset['Fairness_Gap'].mean():+.2f} (paired t={t:.2f}, p={p:.4f} {sig_stars(p)})")

s_gap, f_gap = by_role("Fairness_Gap")
if len(s_gap) and len(f_gap):
    t, p = stats.ttest_ind(s_gap, f_gap, nan_policy="omit")
    print(f"\nDo students and faculty differ in the SIZE of their own double standard?")
    print(f"Student gap {s_gap.mean():+.2f} vs Faculty gap {f_gap.mean():+.2f}: t={t:.2f}, p={p:.4f} {sig_stars(p)}")


# ---------------------------------------------------------------------
# 8. POLICY CONFUSION & ETHICS CONCERN, BY ROLE
# ---------------------------------------------------------------------

print_header("PERCEIVED LACK OF INSTITUTIONAL GUIDANCE")
for label, col in [
    ("Overall confusion (5-item composite)", "Policy_Confusion_Score"),
    ("'No clear guidance from universities' (single item)", COL_NO_GUIDANCE_ITEM),
]:
    s_vals, f_vals = by_role(col)
    t, p = stats.ttest_ind(s_vals, f_vals, equal_var=False, nan_policy="omit")
    print(f"\n{label}:")
    print(f"  Student M={s_vals.mean():.2f}/5, Faculty M={f_vals.mean():.2f}/5, diff={s_vals.mean() - f_vals.mean():+.2f}")
    print(f"  t={t:.2f}, p={p:.4f} {sig_stars(p)}")

print_header("FEAR OF ETHICAL/INTEGRITY VIOLATIONS")
s_eth, f_eth = by_role(COL_ETHICS_VIOLATION_ITEM)
t, p = stats.ttest_ind(s_eth, f_eth, nan_policy="omit")
print(f"Student M={s_eth.mean():.2f}/5, Faculty M={f_eth.mean():.2f}/5")
print(f"t={t:.2f}, p={p:.4f} {sig_stars(p)}")


# ---------------------------------------------------------------------
# 9. ITEM-LEVEL ACCURACY (which item was hardest to call?)
# ---------------------------------------------------------------------

print_header("ITEM-LEVEL ACCURACY")
item_acc = {}
for col in COL_IMAGE_ITEMS:
    item_acc[f"Image: {col}"] = image_scores[col].mean()
for col in COL_TEXT_ITEMS:
    label = col.split(")")[0] + ")"
    item_acc[f"Text {label}"] = text_scores[col].mean()

for name, acc in item_acc.items():
    short_name = name if len(name) <= 60 else name[:57] + "..."
    print(f"  {short_name:<62} {acc:.1%}")

hardest = min(item_acc, key=item_acc.get)
print(f"\nMost frequently misidentified item: {hardest[:60]} ({item_acc[hardest]:.1%} accuracy)")


# ---------------------------------------------------------------------
# 10. DUNNING-KRUGER / CALIBRATION
# ---------------------------------------------------------------------

print_header("CONFIDENCE-ACCURACY CALIBRATION (DUNNING-KRUGER CHECK)")
paired = df[["Pre_Conf_Norm", "Total_Accuracy"]].dropna()
r, p = stats.pearsonr(paired["Pre_Conf_Norm"], paired["Total_Accuracy"])
print(f"Pre-assessment confidence vs actual accuracy: r={r:.3f}, p={p:.3f} {sig_stars(p)}")
print("(A near-zero or negative r means confidence going in didn't predict actual performance.)")

fac_df = df[df["Role_Clean"] == "Faculty"][["Pre_Conf_Norm", "Total_Accuracy"]].dropna()
if len(fac_df) > 2:
    r_fac, p_fac = stats.pearsonr(fac_df["Pre_Conf_Norm"], fac_df["Total_Accuracy"])
    print(f"Faculty-only: r={r_fac:.3f}, p={p_fac:.3f} {sig_stars(p_fac)}")


# ---------------------------------------------------------------------
# 11. FULL SWEEP: EVERY VARIABLE COMPARED BY ROLE, RANKED BY P-VALUE
# ---------------------------------------------------------------------

print_header("FULL SWEEP: STUDENT VS FACULTY ON EVERY SCALE, RANKED BY SIGNIFICANCE")

sweep_targets = (
    [COL_PRE_CONF_TEXT, COL_PRE_CONF_IMAGE, COL_POST_CONF_TEXT, COL_POST_CONF_IMAGE]
    + COL_GUIDELINES
    + COL_PLAGIARISM
    + COL_SKILL_LOSS
    + [
        "Fairness_Gap",
        "Policy_Confusion_Score",
        "Fear_Plagiarism_Score",
        "Fear_Skill_Loss_Score",
        "Concern_Accuracy_Score",
        "Total_Accuracy",
    ]
)

rows = []
for col in sweep_targets:
    s_vals, f_vals = by_role(col)
    if len(s_vals) < 2 or len(f_vals) < 2:
        continue
    t, p = stats.ttest_ind(s_vals, f_vals, equal_var=False)
    label = col if len(col) <= 55 else col[:52] + "..."
    rows.append(
        {
            "Variable": label,
            "Student": round(s_vals.mean(), 2),
            "Faculty": round(f_vals.mean(), 2),
            "Diff": round(s_vals.mean() - f_vals.mean(), 2),
            "p": p,
        }
    )

sweep_df = pd.DataFrame(rows).sort_values("p")
print(f"{'Variable':<58} {'Stud':>6} {'Fac':>6} {'Diff':>7} {'p':>8}  Sig")
print("-" * 92)
for _, row in sweep_df.iterrows():
    print(
        f"{row['Variable']:<58} {row['Student']:>6.2f} {row['Faculty']:>6.2f} "
        f"{row['Diff']:>+7.2f} {row['p']:>8.4f}  {sig_stars(row['p'])}"
    )

print("\nDone.")
