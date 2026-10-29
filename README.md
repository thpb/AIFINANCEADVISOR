# Expense Categorizer

A private, local finance assistant that measures its accuracy, knows when it is unsure, and learns from your corrections.

The rule for every design choice is **code does the maths, the AI only explains**:

- A machine learning model (sentence embeddings + LightGBM) categorizes each transaction and reports how sure it is.
- Every total, average, budget, forecast, savings plan and chart number is calculated by Python code.
- Language models are only used to (a) fill gaps when reading a typed sentence, (b) pick a chart from a fixed menu, and (c) give advice using numbers the app hands them. Every dollar amount the AI writes is checked against the real data.

## What it does

- **Type a sentence**: "spent 50 on groceries yesterday", "paid $12.50 at Starbucks", "got paid 2000 salary". Rules read the amount, the date (always calculated by code, day-first Australian dates) and the merchant. The local AI model only steps in when the rules miss.
- **Categorization you can trust**: the model's confidence is shown. Anything under 60% goes to "Worth a second look".
- **Learns from corrections**: change a category once and the app remembers that merchant and fixes every earlier transaction from it.
- **Budgets** per category: a warning at 80%, an alert over 100%, right after you save a purchase.
- **Income and savings**: savings per month and savings rate.
- **Plan ahead**: a month-end forecast (simple run-rate) and "what if I spent less?" sliders.
- **AI advisor** (local Qwen 2.5 3B through Ollama, or Groq in the cloud): savings plans that add up exactly, charts and tables calculated by code, and a warning when an answer mentions a dollar amount that isn't in your data.
- **Accounts**: several people can use one install. Passwords are stored as salted PBKDF2 hashes and every query is filtered by user.

## Setup (Windows PowerShell)

Python 3.12 or 3.13 is recommended. This project was also built and tested on Python 3.14, where every requirement installed fine.

```powershell
cd expense-categorizer
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If activation is blocked, run this once and try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Run order

```powershell
python src/generate_realistic_data.py   # writes data/transactions.csv (synthetic, deliberately messy)
python src/train_baseline.py            # optional: TF-IDF + Logistic Regression, for comparison
python src/train_improved.py            # optional: v1 LightGBM, for comparison
python src/train_improved_v2.py         # the model the app uses (required)
python src/diagnose_model.py            # optional: confusion matrix and mistakes, for the report
python src/check_categories.py          # shows how the demo sentences get categorized
python src/make_demo_account.py         # optional: demo account (demo / demo1234), see below
streamlit run src/dashboard.py
```

The first run downloads the `all-MiniLM-L6-v2` embedding model (about 90 MB).

Scripts work from any folder: each one starts with `import _root`, which switches to the project root and creates `data/`.

### Results on the original data

Time-based split: the oldest 80% of transactions train the model and the newest 20% test it (predicting new spending from old spending, like real use). To reproduce exactly, generate with `--end 2026-10-29` (without it, the data ends today and the numbers shift a little):

| Model | Accuracy |
|-------|----------|
| Baseline: TF-IDF + Logistic Regression | 75.6% |
| Improved v1: MiniLM embeddings + LightGBM (defaults) | 87.8% |
| Improved v2: v1 + balanced classes, smaller trees, ambiguity flag | 87.0% |

On this data, v2 is not better than v1. v2 is still the app's model because balanced classes help the rare categories, but the comparison is reported as measured. Most mistakes come from shops that sell everything (Amazon, Target, Costco) and from the 4% of rows that are mislabelled on purpose.

### Larger, more Australian data (the app's current model)

`generate_more_data.py` writes `data/transactions_v2.csv`: 4,000 transactions over 12 months with Australian merchants written the way bank statements show them ("COLES 0456 PARRAMATTA NS", "EFTPOS BUNNINGS"), entries typed the way quick add receives them ("uber home"), monthly bills and subscriptions, weekly groceries and more dining on weekends. The original generator and its results are unchanged.

```powershell
python src/generate_more_data.py --end 2026-10-29
python src/train_baseline.py --data data/transactions_v2.csv
python src/train_improved_v2.py --data data/transactions_v2.csv     # the app now uses this model
python src/evaluate_models.py                                       # writes report_results.md
```

`evaluate_models.py` compares both models on the original data, the new data, and **unseen merchants** (10% of shops held out of training completely). See [report_results.md](report_results.md):

| Test | Baseline | v2 | v2 + keyword hints |
|------|----------|----|--------------------|
| Original data, time split | 75.6% | 87.0% | 87.8% |
| New data, time split | 86.5% | 85.6% | 87.4% |
| New data, unseen merchants | 7.3% | 42.7% | 80.2% |

The data is synthetic, so accuracy on a real bank statement is likely to be lower.

### Going back to the original model

The models trained on the original data were backed up to `data/models_backup_original/`. Either copy them back:

```powershell
Copy-Item data\models_backup_original\*.joblib data\
```

or retrain on the original data: `python src/train_improved_v2.py` (with no `--data`, it uses `data/transactions.csv`).

## Demo account

```powershell
python src/make_demo_account.py
```

Then log in as **demo** with password **demo1234**. The script:

- loads `data/transactions_v2.csv` (or `data/transactions.csv` if the larger file doesn't exist) through the real categorization pipeline (model, keyword hints, rules);
- moves the dates forward so the data ends **today**, so "this month" always has spending on demo day;
- keeps every rent, bill and subscription payment and 15% of the everyday spending, because the 4,000-row training file has far more spending than one person would;
- sets budgets for Groceries, Dining, Shopping and Transport, with **Dining close to its limit this month** so the warnings show;
- adds 6 months of fortnightly salary, sized so the demo person saves about 12%.

It is safe to run again: it deletes the old demo account first, so nothing is duplicated. Run it on the morning of the demo so the dates line up with the day. The account exists only in `data/app.db` on your computer.

Before presenting, `python src/advisor_check.py` asks the advisor five typical questions through the real local model and prints the answers, plans, charts and any number-check warnings. It also warms the model up.

## Using your own bank CSV

Overview page, "More ways to add", "Import a CSV". The file needs `date, merchant, amount` columns. Dates can be day-first (`05/10/2026`) or ISO (`2026-10-05`). A `category` column is allowed but ignored, because the app categorizes every row itself. Importing the same file twice in one session does not add the rows again.

## AI engine

### Local (default, private)

Install [Ollama](https://ollama.com), then:

```powershell
ollama pull qwen2.5:3b-instruct-q8_0
```

Nothing leaves your computer. Any other model you have pulled can be typed into the **Local model** box in the sidebar (`ollama list` shows yours). The context size is 8192 tokens by default. To change it:

```powershell
$env:OLLAMA_NUM_CTX="16384"
```

The first question in a conversation takes longest (about 40-60 seconds on an RTX 3050) because the model reads the whole data summary. Ask one warm-up question before a demo.

### Groq (free cloud)

Choose "Groq (free cloud)" in the sidebar and paste a free API key from console.groq.com. The key is kept in memory only. Groq gives smarter answers, but **your spending summary is sent to Groq's servers** when you ask the advisor. You can also set it before starting:

```powershell
$env:ADVISOR_BACKEND="groq"
$env:GROQ_API_KEY="gsk_..."
```

If Groq retires a model, the app asks Groq which models exist and switches automatically.

## Tests

```powershell
python -m unittest discover -s tests      # 170 unit tests, no network, no real model
python -m unittest discover -s apptests   # 23 Streamlit AppTest flows (needs the trained model)
```

AppTest can't type into `st.data_editor`, so the edit/delete logic is covered by `tests/test_income_and_editing.py`. Check editing and deleting a row once by hand in a browser.

## Project structure

```
src/
  _root.py                   switch to the project root and create data/ (imported first by every script)
  generate_realistic_data.py synthetic, deliberately messy transactions (ambiguous shops, refunds, label noise)
  generate_sample_data.py    older, clean-only generator (kept for comparison)
  generate_more_data.py      4,000 rows with Australian merchants, statement noise and typed entries
  evaluate_models.py         baseline vs v2 on original data, new data and unseen merchants
  clean_data.py              clean_merchant(), add_features(), load_and_clean()
  train_baseline.py          TF-IDF + Logistic Regression
  train_improved.py          embeddings + LightGBM v1
  train_improved_v2.py       embeddings + LightGBM v2 (the app's model) and add_ambiguity_feature()
  diagnose_model.py          per-category report, confusion matrix, mistakes
  categorizer.py             the pipeline: model, then keyword hints, then user rules
  category_hints.py          keyword hints for plain words and Australian merchants
  check_categories.py        prints how sample sentences get categorized by the real model
  nl_entry.py                plain-English sentence parser (rules first, AI fallback)
  db.py                      SQLite storage and migrations
  auth.py                    sign up and log in with PBKDF2 password hashing
  budgets.py                 budget maths and money formatting
  corrections.py             merchant rules (learning from corrections) and low-confidence review
  planning.py                month-end forecast and the what-if saver
  savings.py                 income, savings, savings rate
  table_edits.py             turns st.data_editor changes into database updates
  money_facts.py             numbers for the advisor, savings-goal plans, the number check
  advisor.py                 Ollama/Groq calls, prompts, conversation memory
  advisor_charts.py          charts and tables the advisor can request, calculated by code
  make_demo_account.py       demo account (demo / demo1234) for the live demo
  advisor_check.py           asks the advisor five questions through the real model, without the browser
  ui.py                      CSS, colours, chart functions, small HTML helpers
  dashboard.py               the Streamlit app
tests/                       unit tests (fakes and mocks only)
apptests/                    Streamlit AppTest flows through the real app
```

## Known limits

- The model learned from US bank-style merchant names, so plain words and Australian shops rely on the keyword hints.
- The forecast is a simple run-rate. It can't know about bills that haven't arrived yet.
- A 3B local model sometimes writes a number it shouldn't. The app can't stop that, but it flags it under the answer.
- The sidebar AI engine settings apply to the whole running app, which suits one person on one computer.

## Acknowledgements

Ideas (no code) from two MIT-licensed projects:

- [Spritan/expense_tracker](https://github.com/Spritan/expense_tracker): typed-sentence entry and income.
- [nirban256/expense_tracker](https://github.com/nirban256/expense_tracker): category budgets.

All code in this project is original. sugapriya-k/Final-Year-Project was reviewed but not used because it has no licence.
