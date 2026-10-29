# Progress

Spec: `../claude_code_prompt_expense_categorizer.md` (phases in section 15).
If resuming in a new session: read the spec and this file, then continue from the next unfinished step.

## Phases

| # | Phase | Status |
|---|-------|--------|
| 1 | Skeleton: folders, `_root.py`, requirements, `.gitignore`, `data/.gitkeep` | Done |
| 2 | Data generator, cleaning, baseline + v1 + v2 training, diagnose script | Done |
| 3 | `db.py` + `auth.py` + tests (incl. migrations) | Done |
| 4 | `category_hints.py`, `corrections.py`, `categorizer.py`, `check_categories.py` + tests | Done |
| 5 | `nl_entry.py` + tests | Done |
| 6 | `budgets.py`, `planning.py`, `savings.py`, `table_edits.py`, `money_facts.py` + tests | Done |
| 7 | `advisor.py`, `advisor_charts.py` + tests (network mocked) | Done |
| 8 | `ui.py` + `dashboard.py` | Done |
| 9 | AppTest scripts, manual browser pass | **In progress** |
| 10 | README, final test run, zip | Not started |

## Where phase 9 is

- Done: `apptests/test_app.py` (10 AppTest flows: every page, login gate, sentence + hint caption, correction
  learns a rule, income sentence + Saved stat, Expense/Income radio switch, budget toast after saving,
  forecast page, advisor savings plan, advisor chart + number-check warning, Ollama unreachable). All pass.
- Next: manual browser pass (light/dark, phone width, no clipped title, no garbled $, edit/delete in each editor).

## Test status

`python -m unittest discover -s tests`: 133 tests, all passing.
`python -m unittest discover -s apptests`: 10 AppTest flows, all passing (needs the trained model).

## Measured results (generated data, time-based 80/20 split, generated 2026-10-29)

- Baseline (TF-IDF + LogReg): 75.6% (spec reference ~73%)
- Improved v1 (MiniLM + LightGBM defaults): 87.8%
- Improved v2 (the app's model): 87.0% (spec reference ~82%)
- The generator uses `date.today()`, so exact numbers shift slightly depending on the day it is run.

## Decisions and deviations from the spec

- **Python 3.14** is used (only version installed). All requirements installed fine on 3.14
  (torch 2.15, lightgbm 4.7, streamlit 1.66.1, plotly 7.1, pandas 3.1).
- Project lives in `D:\Ai Finance Advisor\expense-categorizer\`; venv folder is `venv/`.
- v1 scored slightly above v2 on this generated data (87.8% vs 87.0%). Reported honestly, not tuned to hide it.
- `categorize()` also stores `model_confidence` (the model's own score before hints/rules), used by
  `check_categories.py` for the "sure" column.
- HTML blocks (headline, stats, budget rows) use `st.html` instead of `st.markdown(unsafe_allow_html=True)`:
  `st.html` does no Markdown/LaTeX processing, so `$` amounts are always safe there.
- `savings.monthly_table` keeps `rate` as an object column so "no income" is `None`, not NaN.
- Month-name dates: "october 5th" (month first) is tried before "5 oct" so an amount right before a month
  name ("coffee 5 october 5th") is not read as the day.
- CSV import ignores any `category` column and categorizes every row itself; ISO dates and day-first dates
  are both accepted.
- The sidebar engine/model/key settings are written to `os.environ` (as the spec says), so they are
  process-wide. Fine for a local single-user app.
