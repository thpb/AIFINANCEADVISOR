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
| 9 | AppTest scripts, manual browser pass | Done (except hand-editing a grid cell, see below) |
| 10 | README, final test run, zip | **In progress** |

## Phase 9 results

- `apptests/test_app.py`: 11 AppTest flows, all passing (adds: a follow-up answer may quote an earlier plan).
- Browser pass with the real `qwen2.5:3b-instruct-q8_0` on Ollama: login, Overview, Plan ahead, Transactions,
  AI advisor; dark and light mode; 375px phone width (no horizontal scroll, title not clipped); no garbled `$`.
- Forecast checked by hand: $3,951.44 x 31/29 = $4,224; $4,800 - $4,224 = $576 (12%). Matches the app.
- Fixed during the pass: clipped bar-chart category labels (axis automargin), squashed button rows (now
  horizontal containers), the 3B model dropping `categories` from chart requests (code now adds categories
  the question names), and the number check flagging amounts from an earlier savings plan (references now
  include every app note in the conversation).
- **Not done by me**: editing/deleting a cell in `st.data_editor` by hand. The grid is canvas-based and
  synthetic clicks didn't drive it reliably. The logic is covered by `tests/test_income_and_editing.py`;
  someone should check it once in a browser.

## Where phase 10 is

- Next: README, final full test run, zip without `data/` contents, `venv` or `__pycache__`.

## Test status

`python -m unittest discover -s tests`: 134 tests, all passing.
`python -m unittest discover -s apptests`: 11 AppTest flows, all passing (needs the trained model).

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
- Added `src/seed_demo.py` (not in the spec): creates a local demo account (demo / demo123) loaded with the
  generated data, a fortnightly salary and budgets, for the live demo.
- Number check references = the data summary + every user message with its app note in the conversation
  (the spec said "the question with its app note"); otherwise follow-ups quoting an earlier plan get flagged.
- `pick_visual` adds category names spelled out in the question when the model leaves `categories` empty.
- Streamlit's file watcher did not reload the edited `advisor` module during the pass; restart
  `streamlit run` after changing code in `src/` other than `dashboard.py`.
- The first advisor question in a conversation takes about 40-60 seconds on the RTX 3050 (the model reads
  the whole data summary); later ones are faster. Worth warming up before a live demo.
