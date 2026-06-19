# Persian AI-Generated Text Detection

> Code, experiments, and results accompanying the paper *"ارزیابی تجربی تشخیص متن تولیدشده توسط مدل‌های زبانی بزرگ در زبان فارسی: تحلیل تعمیم‌پذیری بین‌مولدی و مقاومت در برابر بازنویسی"* (ICMCAI 2026).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Dataset](https://img.shields.io/badge/dataset-persian--ai--text--corpus-green.svg)](https://github.com/asrinvakili/persian-ai-text-corpus)

[فارسی](#فارسی) | [English](#english)

---

## فارسی

این مخزن کد، نوت‌بوک‌ها، نتایج و نمودارهای پژوهش «تشخیص متن تولیدشده توسط مدل‌های زبانی بزرگ در زبان فارسی» را نگه می‌دارد. پژوهش روی پیکره موازی [`persian-ai-text-corpus`](https://github.com/asrinvakili/persian-ai-text-corpus) اجرا شده است.

### سؤالات پژوهش

1. آیا یک طبقه‌بند مبتنی بر ParsBERT می‌تواند متن فارسی تولیدشده توسط LLM را تشخیص دهد؟
2. آیا طبقه‌بند آموزش‌دیده روی یک مولد، به مولدهای دیده‌نشده تعمیم می‌یابد؟
3. آیا چنین طبقه‌بندی در برابر بازنویسی متن از طریق ترجمه برگشتی پایدار است؟

### یافته‌ها در یک نگاه

| شرایط | دقت |
|---|---|
| baseline (آموزش روی هر سه مولد) | **۹۷٫۷۵٪** |
| آموزش تک‌مولدی (میانگین) | ۹۴٫۶٪ |
| آموزش تک‌مولدی، بدترین حالت (GPT روی DeepSeek) | **۸۲٫۵٪** ← افت ۱۷٫۵ واحد |
| آموزش دومولدی (میانگین) | ۹۷٫۸٪ |
| پس از بازنویسی با ترجمه برگشتی | **۴۹٫۵٪** ← افت ۴۸ واحد |
| دقت روی AI پس از بازنویسی | **۳۵٪** ← پایین‌تر از تصادفی |

### ساختار مخزن

```
persian-ai-text-detection/
├── dataset_builder/        # کد ساخت دیتاست از صفر
│   ├── extract_wikipedia.py
│   ├── generate_ai.py
│   └── build_dataset.py
├── experiments/            # شش نوت‌بوک Colab
│   ├── 01_baseline.ipynb
│   ├── 02_error_analysis.ipynb
│   ├── 03_train_on_one.ipynb
│   ├── 04_train_on_two.ipynb
│   ├── 05_robustness_paraphrasing.ipynb
│   └── 06_make_figures.ipynb
├── results/                # خروجی JSON همه آزمایش‌ها
├── figures/                # پنج نمودار (PNG + PDF)
├── paper/                  # مقاله نهایی (DOCX + PDF)
├── docs/datasheet.md       # مستندات دیتاست
└── ...
```

### بازتولید

```bash
# نصب وابستگی‌ها
pip install -r requirements.txt

# گزینه ۱: استفاده از داده آماده
git clone https://github.com/asrinvakili/persian-ai-text-corpus
# سپس نوت‌بوک‌های experiments/ را به ترتیب در Colab اجرا کنید

# گزینه ۲: ساخت داده از صفر
python dataset_builder/extract_wikipedia.py
python dataset_builder/generate_ai.py
python dataset_builder/build_dataset.py
```

برای راهنمای دقیق گام‌به‌گام، [`RUN.md`](RUN.md) را ببینید.

### نوت‌بوک‌ها — ترتیب اجرا

| # | نوت‌بوک | هدف | زمان روی T4 |
|---|---|---|---|
| 1 | `01_baseline.ipynb` | تنظیم دقیق ParsBERT روی کل داده | ~۲۵ دقیقه |
| 2 | `02_error_analysis.ipynb` | تحلیل خطا به تفکیک منبع/حوزه/طول | ~۱ دقیقه |
| 3 | `03_train_on_one.ipynb` | آموزش با یک مولد، آزمون روی بقیه | ~۴۵ دقیقه |
| 4 | `04_train_on_two.ipynb` | آموزش دومولدی leave-one-out | ~۴۵ دقیقه |
| 5 | `05_robustness_paraphrasing.ipynb` | حمله بازنویسی با ترجمه برگشتی | ~۴۵ دقیقه |
| 6 | `06_make_figures.ipynb` | ساخت پنج نمودار مقاله | ~۲ دقیقه |

### نتایج به‌صورت JSON

همه نتایج به‌صورت ماشین‌خوان در `results/` ذخیره شده‌اند تا برای متاآنالیز یا مقایسه بعدی قابل استفاده باشند:

- `baseline_results.json` — متریک‌های کلی baseline
- `error_analysis.json` — دقت به تفکیک سه بُعد
- `train_on_one_matrix.json` — ماتریس ۳×۳ آزمایش تک‌مولدی
- `train_on_two_matrix.json` — نتایج سه ترکیب دومولدی
- `robustness_results.json` — اثر حمله بازنویسی

### استناد

اگر از این کد یا نتایج استفاده کردید، لطفاً به مقاله مرجع استناد کنید:

```bibtex
@inproceedings{vakili2026persianai,
  author    = {Vakili, Asrin},
  title     = {Empirical Evaluation of AI-Generated Text Detection in Persian:
               Cross-Generator Generalization and Robustness to Paraphrasing},
  booktitle = {2nd International Conference on Management, Computer Science,
               and Artificial Intelligence (ICMCAI)},
  year      = {2026},
  address   = {Tehran, Iran}
}
```

### مجوز

- **کد**: MIT
- **مقاله**: CC BY 4.0
- **داده**: به مخزن [`persian-ai-text-corpus`](https://github.com/asrinvakili/persian-ai-text-corpus) مراجعه کنید

### تماس

**اسرین وکیلی** — گروه مهندسی کامپیوتر، دانشگاه ملی مهارت، تهران، ایران.
برای پرسش یا گزارش مشکل، [Issue](https://github.com/asrinvakili/persian-ai-text-detection/issues) باز کنید.

---

## English

This repository contains the code, notebooks, results, and figures accompanying the paper *"Empirical Evaluation of AI-Generated Text Detection in Persian: Cross-Generator Generalization and Robustness to Paraphrasing"*, presented at ICMCAI 2026.

### Research questions

1. Can a ParsBERT-based classifier reliably detect Persian LLM-generated text?
2. Does a classifier trained on a single generator transfer to unseen generators?
3. Is such a classifier robust to a simple paraphrasing attack via back-translation?

### Key findings

| Condition | Accuracy |
|---|---|
| Baseline (trained on all 3 generators) | **97.75%** |
| Train on one generator (average) | 94.6% |
| Train on one, worst case (GPT → DeepSeek) | **82.5%** (−17.5 pp) |
| Train on two (LOO average) | 97.8% |
| After back-translation paraphrasing | **49.5%** (−48 pp) |
| AI accuracy after paraphrasing | **35%** (below chance) |

### Repository structure

```
persian-ai-text-detection/
├── dataset_builder/        # scripts to build the corpus from scratch
├── experiments/            # 6 Colab notebooks
├── results/                # JSON outputs of all experiments
├── figures/                # 5 figures (PNG + PDF)
├── paper/                  # final paper (DOCX + PDF)
├── docs/datasheet.md       # dataset documentation
└── ...
```

### Reproducing

```bash
pip install -r requirements.txt

# Option 1: use prebuilt corpus
git clone https://github.com/asrinvakili/persian-ai-text-corpus
# Then run experiments/ notebooks in Colab, in order

# Option 2: build corpus from scratch
python dataset_builder/extract_wikipedia.py
python dataset_builder/generate_ai.py
python dataset_builder/build_dataset.py
```

See [`RUN.md`](RUN.md) for the full walkthrough.

### Citation

```bibtex
@inproceedings{vakili2026persianai,
  author    = {Vakili, Asrin},
  title     = {Empirical Evaluation of AI-Generated Text Detection in Persian:
               Cross-Generator Generalization and Robustness to Paraphrasing},
  booktitle = {2nd International Conference on Management, Computer Science,
               and Artificial Intelligence (ICMCAI)},
  year      = {2026},
  address   = {Tehran, Iran}
}
```

### License

Code: MIT. Paper: CC BY 4.0. Dataset: see [`persian-ai-text-corpus`](https://github.com/asrinvakili/persian-ai-text-corpus).

### Contact

**Asrin Vakili** — Department of Computer Engineering, National University of Skills, Tehran, Iran.
Open an [Issue](https://github.com/asrinvakili/persian-ai-text-detection/issues) for questions.
