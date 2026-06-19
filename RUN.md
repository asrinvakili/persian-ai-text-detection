# راهنمای اجرای گام‌به‌گام

این فایل دو سناریوی متفاوت را پوشش می‌دهد: استفاده از داده آماده، یا ساخت داده از صفر.

---

## پیش‌نیازها

- Python 3.10+
- Google Colab با T4 GPU (یا یک GPU محلی با ≥ ۸ گیگ حافظه)
- حساب Google Drive با حداقل ۵ گیگابایت فضای آزاد
- (فقط برای ساخت داده) کلید API از AvalAI

---

## سناریوی الف — بازتولید آزمایش‌ها با داده آماده

این مسیر سریع‌تر است. تنها داده موجود را دانلود می‌کنید و نوت‌بوک‌ها را اجرا.

### گام ۱: دانلود پیکره

```bash
git clone https://github.com/asrinvakili/persian-ai-text-corpus
```

### گام ۲: انتقال داده به Google Drive

```
MyDrive/persian-ai-research/
└── data/
    └── splits/
        ├── train.jsonl
        ├── val.jsonl
        └── test.jsonl
```

می‌توانید فولدر `data/splits/` از مخزن پیکره را مستقیم به این مسیر کپی کنید.

### گام ۳: اجرای نوت‌بوک‌ها در Colab به ترتیب

| # | نوت‌بوک | پیش‌نیاز قبلی | خروجی روی Drive |
|---|---|---|---|
| ۱ | `01_baseline.ipynb` | ندارد | `models/baseline_model/`, `results/baseline_results.json`, `results/baseline_predictions.jsonl` |
| ۲ | `02_error_analysis.ipynb` | نوت‌بوک ۱ | `results/error_analysis.json` |
| ۳ | `03_train_on_one.ipynb` | ندارد (مستقل) | `ablation_train_on_one/`, `results/train_on_one_matrix.json` |
| ۴ | `04_train_on_two.ipynb` | ندارد (مستقل) | `ablation_train_on_two/`, `results/train_on_two_matrix.json` |
| ۵ | `05_robustness_paraphrasing.ipynb` | نوت‌بوک ۱ | `robustness/test_paraphrased.jsonl`, `results/robustness_results.json` |
| ۶ | `06_make_figures.ipynb` | همه نوت‌بوک‌های ۱ تا ۵ | `figures/fig{1,2,3,4,5}.{png,pdf}` |

**زمان کل تقریبی روی T4:** ۳ تا ۴ ساعت.

### نکته مهم — مقاومت در برابر قطع سشن

همه نوت‌بوک‌ها قابلیت ادامه از وقفه را دارند. اگر Colab قطع شد:

1. دوباره به Colab بروید
2. همان نوت‌بوک را باز کنید
3. Runtime → Restart runtime
4. سلول‌ها را از اول اجرا کنید — کد خودکار از آخرین نقطه‌ای که داشتید ادامه می‌دهد

این بدین معناست که فضای Drive باید آزاد باشد. **بررسی کنید Drive شما حداقل ۲ گیگابایت فضای خالی دارد** قبل از اجرای نوت‌بوک‌های ۳، ۴، ۵.

---

## سناریوی ب — ساخت دیتاست از صفر

این مسیر کندتر است (~۴ ساعت) ولی به شما کنترل کامل می‌دهد.

### گام ۱: کلید API

```bash
echo "YOUR_AVALAI_KEY" > api_key.txt
```

### گام ۲: استخراج متن انسانی از ویکی‌پدیا

```bash
python dataset_builder/extract_wikipedia.py
```

- زمان: ~۳۰ دقیقه
- خروجی: `data/human_texts.jsonl` (۱۰۰۰ رکورد)
- نیاز به اینترنت برای بازیابی نسخه‌های پیش از ۲۰۲۰ از ویکی‌پدیا

### گام ۳: تولید متن AI با سه مولد

```bash
python dataset_builder/generate_ai.py
```

- زمان: ~۳ ساعت (سریال) یا ~۱ ساعت (موازی)
- خروجی: `data/ai_texts.jsonl` (۳۰۰۰ رکورد)
- مصرف API: حدود ۳۰۰۰ تماس

### گام ۴: ساخت تقسیم‌بندی نهایی

```bash
python dataset_builder/build_dataset.py
```

- خروجی: `data/splits/{train,val,test}.jsonl`
- تقسیم بر اساس شناسه موضوع، نه نمونه (جلوگیری از نشت داده)

### گام ۵: ادامه با سناریوی الف از گام ۲

---

## رفع اشکال

**خطای «۴۰۳ Forbidden» موقع تماس با AvalAI:**
- اعتبار کلید را چک کنید
- در صورت محدودیت نرخ، بین تماس‌ها فاصله بگذارید

**خطای «No space left on device» در Colab:**
- Drive پر است. فولدرهای checkpoint قدیمی را حذف کنید
- یا از Drive با اشتراک بیشتر استفاده کنید

**نوت‌بوک ۵ متن بازنویسی نمی‌سازد:**
- نام مدل‌های parsinlu را چک کنید — یکی opus دارد و دیگری ندارد (در نوت‌بوک توضیح داده شده)

**خطای «model not found» موقع بارگذاری baseline:**
- ابتدا نوت‌بوک ۱ را اجرا کنید تا `models/baseline_model/` ساخته شود
- یا checkpoint آخر را به مسیر `models/baseline_model/` کپی کنید
