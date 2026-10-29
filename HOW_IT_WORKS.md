# How Expense Categorizer works

A plain-English guide for questions about the project. Each section says which file and function does the
work, so you can open the code and show it.

The rule behind everything: **code does the maths, the AI only explains.**

---

## 1. How a transaction gets its category

Three steps, one after the other. If a later step has an answer, it wins.

**Step 1: the machine learning model makes a guess.** It looks at four kinds of clue:
- the *meaning* of the shop's name. A small language model (MiniLM) turns the name into 384 numbers, so
  "Blue Bottle Coffee" and "Bean There Espresso" end up close together even though they share no words;
- the amount ($6 looks like coffee, $2,150 looks like rent);
- the date (day of the week, day of the month, month);
- two yes/no flags: does this exact payment repeat (a subscription?), and is it a shop that sells
  everything (Amazon, Target, PayPal)?

A LightGBM model (hundreds of small decision trees voting together) turns these clues into a category
and says how sure it is, for example "Dining, 92% sure".
File: `src/categorizer.py`, function `categorize`. The model is trained in `src/train_improved_v2.py`.

**Step 2: keyword hints fix the obvious cases.** The model learned from bank-statement names, so plain
words people type ("coffee", "rent") and Australian shops it never saw ("Coles", "Opal", "maccas") are
only guesses for it. A short, readable list of words fixes those: "coffee" means Dining, "opal" means
Transport. The longest matching phrase wins, so "uber eats" is Dining and "uber" is Transport. Amazon and
Target are deliberately *not* on the list: the model's uncertainty about them is the honest answer.
File: `src/category_hints.py`, functions `find_hint` and `apply_hints`.

**Step 3: your own corrections always win.** When you change a category, the app saves a rule for that
shop. The shop name is cleaned first, so "SQ *BLUE BOTTLE #12" and "blue bottle" count as the same shop.
The app then fixes every earlier transaction from that shop and uses the rule for every future one.
Files: `src/corrections.py`, functions `merchant_key` and `apply_rules`; saving the rule happens in
`src/dashboard.py`, function `learn`.

Anything the model was less than 60% sure about is listed under Transactions, "Worth a second look"
(`corrections.needs_review`). One honest limit: on the test data the model is often over-confident when
it is wrong, so this list catches only some mistakes. The hints and your corrections do most of the
fixing (see `report_results.md`).

---

## 2. How quick add reads a sentence, and why dates are calculated by code

When you type "spent 50 on groceries yesterday", simple rules read it in this order:

1. **Date.** The rules look for date words: "today", "yesterday", "3 days ago", "last friday",
   "12/10" (day first, Australian style), "5 oct". The date is then *calculated* from today's date.
   A date like "25/12" with no year that would land in the future is moved back a year.
   File: `src/nl_entry.py`, function `find_date`.
2. **Amount.** The rules look for numbers like "50", "$12.50", "1,200" or "50 bucks". If there are two
   numbers ("2 coffees 9"), the one that looks like money wins and "2 coffees" stays as the description.
   Function `find_amount`.
3. **Merchant.** Whatever is left, minus filler words ("spent", "on", "at", "the"), is the merchant:
   "groceries". Function `_merchant_from`.
4. **Expense or income.** Words like "salary", "got paid" or "bonus" mean income. Plain "paid" does not
   ("paid $12.50 at Starbucks" is an expense). Function `looks_like_income`.

Only if the rules can't find an amount or a merchant ("thirty bucks on a taxi") does the local AI model
help (`llm_extract`). It may only fill in the amount, the merchant and the date *words* it saw, such as
"last friday". Those words go back through the same `find_date` rules.

**Why dates are calculated by code:** a language model doesn't reliably know today's date, and it is bad
at counting days ("last Friday" from a Thursday is 6 days back). One wrong date puts a purchase in the
wrong month and changes every budget and total. Code is right every time and can be tested: about 35
tests cover the sentence reader (`tests/test_nl_entry.py`).

Before anything is saved, a "Check this before it's saved" box shows what was read and why the category
was chosen, so you can fix it.

---

## 3. How the savings plan is calculated, and why the AI doesn't calculate it

Question: "How can I save $300 this month?"

1. Code spots the goal: a saving word ("save", "cut", "put aside") followed by an amount. "Cut dining by
   20%" and "save in 2026" are not goals. File: `src/money_facts.py`, function `detect_goal`.
2. Code works out your **normal month**: the average of your last 3 complete months, per category
   (`src/planning.py`, `monthly_average`).
3. **Bills are left out**, because rent and power can't be cut this month.
4. Every other category is cut by **the same percentage**, so bigger categories give more dollars.
   Example: if your flexible spending is $2,761.50 a month, $300 is 10.9% of it, so Shopping
   ($1,274.87) gives $138.50, Groceries ($921.34) gives $100.09, and so on.
5. **No category is cut by more than 30%.** If the goal is bigger than that allows, the plan says so and
   shows the most that's realistic.
6. Rounding to cents can make the total a cent out, so the leftover cent goes on the biggest category.
   The plan always adds up to **exactly** the goal.
   Function: `savings_goal`.

The app shows the plan as a table and gives the AI only the result ("Groceries: spend $79.79 less this
month"). The AI's job is to add practical tips. File: `src/advisor.py`, function `answer_question`.

**Why the AI doesn't calculate it:** small language models predict words; they don't do arithmetic. In
testing, a 3B model once "saved" $818.64 from a $300 goal. When it was given several numbers per line,
it sometimes quoted the wrong one. So the app gives it one number per category, already calculated.

The same idea runs through the rest of the advisor. The app ranks "what changed most" itself, answers
budget questions from its own budget table, and adds a "Change" column to monthly charts, so the model
never has to subtract.

---

## 4. How the number check catches invented numbers

After the AI answers, code reads every dollar amount and percentage in the answer and compares it with
the numbers the app actually gave the AI: the data summary and the app notes in the conversation.

- A dollar amount passes if it is within $1 or 1% of a real number (so "$1,275" for $1,274.87 is fine).
- A percentage passes if it is within one point of a real percentage.
- Anything else gets a yellow warning under the answer: "This answer mentions $818.64, which doesn't
  come from your data. Check it before relying on it."

File: `src/money_facts.py`, function `unverified_amounts`. The warning is shown in `src/dashboard.py`,
function `page_advisor`.

**Its limit:** it checks *that* a number is real, not that it is used correctly. If the AI swaps a
"from" and a "to", both numbers are real and pass. That's why the app also gives the AI as few numbers
as possible for each question (section 3).

---

## 5. What quantization is, and why Qwen 2.5 3B in 8-bit with an 8K context

**What a model is made of:** a language model is billions of numbers ("weights"). Qwen 2.5 3B has about
3 billion of them. Normally each weight takes 16 bits, which for 3 billion weights is about 6 GB.

**Quantization** stores each weight with fewer bits, like saving a photo as a smaller JPEG.
- **8-bit** (the `q8_0` in `qwen2.5:3b-instruct-q8_0`): each weight takes 8 bits, so the model is about
  3.3 GB. Answers are almost the same as the full-size model.
- **4-bit**: about 2 GB and faster, but the model makes more mistakes, and in a finance app mistakes
  with numbers matter.

8-bit is the middle choice: small enough for a laptop graphics card (RTX 3050) and 16 GB of RAM, while
keeping quality close to the original. A 3B model is small enough to answer in seconds on this computer
and runs **completely offline**, so your spending never leaves the computer.

**Context** is how much text the model can read at once, measured in tokens (pieces of words). The
app's data summary is about 3,700 tokens: numbers take more tokens than words, about 2 characters per
token. An 8K (8,192-token) window fits the summary, the recent conversation and the answer. A bigger
window would use more memory and be slower. Ollama silently cuts the *oldest* text when a request is too
long, and the oldest text is the spending data. So the app trims old questions first and always keeps
the data (`src/advisor.py`, `_trim_history`; `num_ctx` is sent on every call in `ollama_chat`).

Setting: `DEFAULT_LOCAL_MODEL`, `LOCAL_CONTEXT` and `ADVICE_TEMPERATURE = 0.2` (low randomness, so
answers stay close to the facts) in `src/advisor.py`. The model name can be changed in the sidebar.

---

## 6. Why the forecast is a simple run-rate

"On pace for $4,224 by the end of October" means: spending so far, divided by the days so far, times
the days in the month. Example: $3,951.44 in 29 days, so $3,951.44 / 29 x 31 = $4,224.
File: `src/planning.py`, function `forecast_month`.

Why so simple:
- **You can check it with a calculator.** In a live demo or a Q&A, every number can be explained in one
  sentence. That matters more than a few dollars of extra accuracy.
- **There isn't enough data for anything smarter.** Models that learn seasons or paydays need years of
  history from the same person. One person has a few months.
- **It is honest about its limits.** The app says so on screen: "This is a simple run-rate, so it assumes
  you keep spending at the same pace. It can't know about bills that haven't been paid yet." In the first
  4 days of a month it doesn't forecast at all, because a single big shop would make the estimate swing
  wildly.

Budget warnings use the forecast too: "Dining is on pace for $465 by month end, $65 over its $400
budget" (`forecast_messages`).
