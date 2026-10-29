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
| 10 | README, final test run, zip | Done |

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

## Phase 10 results

- README written (setup, run order, measured accuracies, CSV import, AI engines, tests, structure, credits).
- Final run: 134 unit tests and 11 AppTest flows, all passing.
- Zip: `../expense-categorizer.zip`, built with `git archive` (tracked files only: no `data/` contents,
  `venv` or `__pycache__`).


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

# Improvement prompts (`../claude_code_improvement_prompts.md`)

| # | Prompt | Status |
|---|--------|--------|
| 0 | CLAUDE.md project memory | Done |
| 1 | Acceptance checklist, PASS/FAIL | Done |
| 2 | More realistic training data | Done |
| 3 | Check categories with the real model | Next |
| 4 | Test the advisor with real Ollama | Not started |
| 5 | Safe demo account | Not started |
| 6 | Nothing crashes during the live demo | Not started |
| 7 | Report material | Not started |
| 8 | HOW_IT_WORKS.md for Q&A | Not started |
| 9 | Final tidy-up, "Final submission" commit | Not started |

## Notes from the improvement prompts

- Prompt 0 asks to keep "baseline 73.0%, v2 81.9%" reproducible. Those are the spec's reference figures, not
  results of this generator. The generator also used `date.today()`, so even our own numbers moved daily.
  Added an optional `--end YYYY-MM-DD` to `generate_realistic_data.py` (default unchanged; verified
  byte-identical output for `--end 2026-10-29`). Reported results: 75.6% / 87.0% with `--end 2026-10-29`.

## Prompt 1: acceptance checklist (29/10/2026)

| # | Item | Result | How it was checked |
|---|------|--------|--------------------|
| 1 | Fresh Windows setup from README only | PASS | `git clone` to a scratch folder, new venv, `pip install -r requirements.txt`, generate, train, unit tests |
| 2 | Baseline ~73%, v2 ~82% | FAIL vs reference | Measured 75.6% / 87.0% (reproducible with `--end 2026-10-29`). Not "fixed": changing the data to hit a target would be dishonest |
| 3 | "spent 50 on groceries yesterday" | PASS | AppTest `test_sentence_hint_then_correction_learns_rule` + browser |
| 4 | One correction fixes past and future rows | PASS | New AppTest `test_correction_fixes_past_and_future` |
| 5 | Low-confidence rows in review, gone when corrected | PASS | Same AppTest: tab goes from (1) to (0) |
| 6 | Budget warning at 80%, alert over 100%, right after saving | PASS | AppTests for 90% and over-budget toasts; unit tests for exact 80%/100% |
| 7 | Forecast, what-if, savings match hand calc | PASS | Demo data: forecast $4,224 = 3951.44x31/29; Dining 20% = $78.88/month by hand and in app; October saved $848.56 both |
| 8 | "got paid 2000 salary" is income, Saved stat updates | PASS | AppTest `test_income_sentence_and_saved_stat` |
| 9 | Edit and delete in both editors, nothing saved until Save | PASS after fix | Browser: edited an amount and deleted a row (DB unchanged until Save, then saved). **Bug found and fixed**: Discard left the unsaved deletion showing in the grid. Editors now use versioned keys |
| 10 | "$300" plan adds up exactly, max 30%, Bills out | PASS | Unit test (spec reference example) + demo data: total $300.00, max cut 9.8%, no Bills |
| 11 | Invented amounts show the warning | PASS | AppTest with $818.64 + real Qwen flagged an invented $4,951.44 |
| 12 | Dining chart in Dining's colour | PASS after earlier fix | Browser with real Qwen |
| 13 | Model label, Local model box, Groq errors | PASS (Groq with a real key not tested: no key) | New AppTests: Local model box sends `llama3.2:3b`; Groq with no key shows the sidebar message; mocked Groq unit tests |
| 14 | No $ as maths or code box | PASS | New unit test on "$398 in Jan to $1,855 in Aug" + browser |
| 15 | Re-uploading a CSV adds nothing | PASS after fix | **Was only true within one browser session.** Now rows already stored are skipped (`db.only_new`). AppTest uploads the same file in two sessions |
| 16 | Older app.db opens intact | PASS | Unit test builds an old-schema database |
| 17 | All tests pass | PASS | 136 unit tests, 16 AppTests |

## Prompt 2: more realistic training data

- `src/generate_more_data.py` (seed 2026, `--end 2026-10-29`): 4,000 rows over 12 months, Australian merchants
  with statement noise, 5% typed entries, monthly rent/bills/subscriptions, weekly groceries, weekend dining,
  ambiguous shops and transfers, ~4% label noise, 60 refunds. US merchants kept.
- `train_baseline.py`, `train_improved_v2.py`, `diagnose_model.py` take `--data`. Original models backed up to
  `data/models_backup_original/`. **The app's v2 model is now trained on `transactions_v2.csv`** (85.8%).
- `clean_merchant` also strips `EFTPOS ` and `VISA PURCHASE ` prefixes (original data has neither; original
  results re-checked: still 75.6% / 87.0%).
- `src/evaluate_models.py` -> `report_results.md`: original 75.6/87.0, new 86.5/85.6, unseen merchants
  7.3/42.7 (79.9 with hints). On the new data v2 is slightly below the baseline; an experiment without the
  `month` feature moved v2 by only 0.5 points, so the cause is that most test shops are already known.
- 11 generator tests (`tests/test_generate_more_data.py`). 147 unit tests, 16 AppTests pass.
- `check_categories.py` outputs: `data/check_categories_original.txt` and `data/check_categories_v2.txt`.
