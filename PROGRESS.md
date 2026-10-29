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
- The demo account script is now `src/make_demo_account.py` (demo / demo1234, prompt 5); `seed_demo.py` was removed.
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
| 3 | Check categories with the real model | Done |
| 4 | Test the advisor with real Ollama | Done |
| 5 | Safe demo account | Done |
| 6 | Nothing crashes during the live demo | Done |
| 7 | Report material | Done |
| 8 | HOW_IT_WORKS.md for Q&A | Next |
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

## Prompt 3: check categories with the real model

- All 16 demo sentences get the right final category with the v2-data model.
- Extra Australian phrasing checked (30 sentences). Wrong or shaky: "rego" (Other, should be Transport),
  "dentist" (Shopping 41%), "brekkie" (Dining 56%). Added hints: Dining "maccas", "brekkie"; Transport "servo",
  "rego"; new Other list "doctor", "dentist", "chemist", "pharmacy", "haircut", "vet". Model settings unchanged.
- Left to the model on purpose (genuinely debatable): pub, 7-Eleven, Dan Murphy's, and Amazon (by design).

## Prompt 4: advisor with the real qwen2.5:3b-instruct-q8_0

New `src/advisor_check.py` asks the 5 questions through `advisor.answer_question` (the same code path the app
now uses; moved out of `dashboard.py`). Outputs: `data/advisor_check_before.txt`, `data/advisor_check_after.txt`.

| Question | Before | After |
|----------|--------|-------|
| Save $300 | Plan correct. A rerun mixed up the cut with the new spending ("reduce by $734") | Note gives one amount per category; 3/3 runs quote the cuts exactly |
| What changed most | Wrong: said Transport (+$29.55); real answer Groceries (-$406.15). Changes were alphabetical | Changes sorted by code, "Biggest change" named; change questions get ONE comparison (this month so far vs last month); 3/3 correct |
| Dining chart | Right chart, but the model subtracted months itself and got two wrong | Code adds a "Change" column; note forbids calculating; numbers correct |
| Budgets | Called limits "averages", invented $915.52 and $1,100.22 | Deterministic route: app's budget table + month-end forecast, state words exact |
| Top 5 purchases | Model restricted it to February 2026 | Code decides period/top N from the question's words; all months |

Other fixes: the number check now also checks percentages (a run invented "-848%"); model headings
("### Tips") are shown as bold text. 156 unit tests, 16 AppTests pass.
Known limit: the number check can't catch a real number used in the wrong role (e.g. a "from" and "to"
swapped); the narrower notes make that much rarer but not impossible with a 3B model.

## Prompt 5: safe demo account

- `src/make_demo_account.py`: demo / demo1234, real pipeline, `transactions_v2.csv` with fallback, dates shifted
  so the data ends today, budgets for Groceries/Dining/Shopping/Transport (Dining ~90% this month), 6 months of
  fortnightly salary sized for ~12% savings. Rerun deletes the old demo user first (`db.delete_user`).
- Found: the 4,000-row training file is ~$28,000/month of spending (fine for training, absurd for one person).
  The demo keeps every fixed payment and a fixed 15% sample of everyday rows (~$7,300/month).
- 6 tests in `tests/test_make_demo_account.py` (fake models, temp DB). README "Demo account" section.

## Prompt 6: nothing crashes during the live demo

| Case | Result | Test |
|------|--------|------|
| Ollama not running | Friendly error (advisor + quick add) | AppTests `test_advisor_unreachable`, `test_sentence_without_amount_and_model_down` |
| Wrong local model name | **Fixed**: quick add used to say "isn't running"; now says the model isn't installed and how to pull it | `test_wrong_local_model_name`, unit `test_wrong_model_name_says_so` |
| Wrong Groq key | "Groq rejected the API key..." | `test_wrong_groq_key` |
| Groq with no key | "Add your Groq API key..." | `test_groq_without_key_explains` |
| Empty account | Every page renders with an invitation to add data | `test_empty_account_every_page` |
| CSV wrong/missing columns, empty file, bad rows | **Fixed**: bad rows used to fail the whole file with a pandas error; now skipped and counted, plain messages (`clean_data.read_bank_csv`) | `test_bad_csv_files` + 4 unit tests |
| Sentence with no amount | Friendly error with an example | AppTest + unit tests |
| No income / no budgets | "Add income" stat, "No budgets yet" caption, advisor says no budgets are set | `test_no_income_no_budgets_and_empty_month`, unit budget test |
| Month with no spending | "You spent $0 in ..."; Plan ahead "finished: you spent $0" | same AppTest |
| Very large amounts | Quick add capped at $1,000,000 (clear error); **fixed**: the hand-entry, income, confirm and budget boxes had no cap | `test_very_large_amount` |

Also fixed: "1 categories are worth checking" -> "1 category is worth checking".
New `advisor.AdvisorUnreachable` separates "can't reach the engine" from fixable errors.
170 unit tests, 23 AppTests pass.

### Check by hand (could not be tested automatically)
- Groq with a real, valid key (no key here; all Groq paths are mocked).
- Stopping the real Ollama service mid-demo (tested with a mocked connection error).
- Your own bank's CSV: Australian banks often export without a header row or call the columns
  "Description"/"Debit"/"Credit". The import needs `date, merchant, amount` headers; rename them first.
- How the app looks on the presentation screen/projector, in light and dark mode.
- Typing into the data editor grid (checked once by hand on 30/10/2026: edit, delete, Save, Discard all work).
- The first advisor answer after starting Ollama takes 40-60 s; run `python src/advisor_check.py` to warm up.

## Prompt 7: report material

`report_results.md` is fully generated by `python src/evaluate_models.py` (reproducible): accuracy table
(original/new/unseen), v2 per-category P/R/F1, confusion matrix, where the mistakes are (ambiguous shops 43.4%
vs 92.0% other shops vs 92.9% typed), top 10 features (amount, day_of_month, month, day_of_week, then
embedding values; embeddings = 51% of splits), confidence right vs wrong, and two explanation paragraphs.
**Honest finding**: v2 is over-confident on this data (86% average confidence when wrong); the 60% review line
catches only 12% of its mistakes. The report says so and names calibration as a possible improvement.
Hints raise unseen-merchant accuracy to 80.2% (was 79.9% before prompt 3's new hints).
