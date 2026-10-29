# CLAUDE.md

Expense Categorizer: a private, local AI personal finance app (Streamlit + SQLite + LightGBM + Ollama).
Full spec: `../claude_code_prompt_expense_categorizer.md`. Follow-up prompts: `../claude_code_improvement_prompts.md`.
Progress and decisions: `PROGRESS.md` (read it first when resuming; update it and commit after each phase or prompt).

## Rules

- Windows 11 + PowerShell. Activate the venv with `venv\Scripts\Activate.ps1`.
- Core rule: code does the maths, the AI only explains. Never let an LLM calculate numbers.
- Dates are Australian, day-first (dd/mm/yyyy).
- Run `python -m unittest discover -s tests` after every change. All tests must pass before committing.
  The Streamlit flows are in `python -m unittest discover -s apptests` (needs the trained model).
- UI text is plain, short English. No em dashes.
- Keep the original data generator and the reported accuracies reproducible:
  `python src/generate_realistic_data.py --end 2026-10-29`, then `train_baseline.py` gives 75.6% and
  `train_improved_v2.py` gives 87.0%. (The spec's reference figures, 73.0% and 81.9%, came from the spec
  author's own generator and were never measured on this one.)
- Commit after every finished task with a clear message.

## Commands

```powershell
python src/generate_realistic_data.py      # data/transactions.csv
python src/train_improved_v2.py            # data/improved_model_v2.joblib (the app's model)
streamlit run src/dashboard.py
```

## Layout

- `src/`: one module per concern (see README "Project structure"). Every script starts with `import _root`.
- `tests/`: unit tests with fakes and mocks, no network. `tests/helpers.py` gives a throwaway database.
- `apptests/`: Streamlit AppTest flows through the real app with a mocked Ollama.
- `data/`: git-ignored (CSVs, models, `app.db`).
