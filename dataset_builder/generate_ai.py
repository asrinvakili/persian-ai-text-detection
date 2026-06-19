"""
generate_ai.py
==============
تولید متن‌های موازی هوش‌مصنوعی برای هر موضوع انسانی، با سه مدل متفاوت
(deepseek-v3.1، gpt-4o-mini، gemini-2.5-flash-lite) از طریق دروازه AvalAI.

طرح «تطبیقی»: هر موضوع در human_texts.jsonl به همه مدل‌ها داده می‌شود.
نتیجه: ۱۰۰۰ متن انسانی × ۳ مدل = ۳۰۰۰ متن هوش‌مصنوعی.

ویژگی‌ها:
  • لنگر موضوع: جمله اول متن انسانی به‌عنوان موضوع به مدل داده می‌شود تا از
    توهم در نام‌های خاص فارسی جلوگیری شود. سپس اگر مدل آن جمله را کپی کرد،
    حذف می‌شود تا متن انسان و AI هیچ جمله مشترکی نداشته باشند.
  • پردازش گروهی: هر مدل به‌طور کامل تمام موضوعاتش را پردازش می‌کند، بعد
    مدل بعدی شروع می‌کند. اگر مدلی ناپایدار شد، روی آن معطل نمی‌مانیم.
  • شکست‌محور بودن: اگر ۵ متن پشت سر هم از یک مدل شکست خورد، آن مدل
    کنار گذاشته می‌شود و به مدل بعدی می‌رویم.
  • قابلیت ادامه: اگر اجرا قطع شد، با اجرای دوباره دقیقاً از همان‌جا
    ادامه می‌دهد (تشخیص جفت‌های (موضوع، مدل) که قبلاً تولید شده‌اند).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from typing import Iterable

import requests

# ---------------------------------------------------------------------------
# پیکربندی
# ---------------------------------------------------------------------------

API_BASE = "https://api.avalai.ir/v1/chat/completions"
INPUT_PATH = os.environ.get("HUMAN_PATH", "data/human_texts.jsonl")
OUT_PATH = os.environ.get("AI_PATH", "data/ai_texts.jsonl")
KEY_FILE = "api_key.txt"

MODELS = ["deepseek-v3.1", "gpt-4o-mini", "gemini-2.5-flash-lite"]

# نرخ مجاز در دقیقه (RPM) برای محاسبه فاصله بین درخواست‌ها
RPM = {
    "deepseek-v3.1": 250,
    "gpt-4o-mini": 500,
    "gemini-2.5-flash-lite": 50,
    "claude-haiku-4-5": 25,  # برای افزودن احتمالی در آینده نگه‌داشته شده
}

# برچسب کوتاه برای ساخت شناسه یکتا (ai_{human_id}_{tag})
MODEL_TAG = {
    "deepseek-v3.1": "deepseek",
    "gpt-4o-mini": "gpt",
    "gemini-2.5-flash-lite": "gemini",
    "claude-haiku-4-5": "claude",
}

MAX_RETRIES = 3
REQUEST_TIMEOUT = 90
FAIL_STREAK_LIMIT = 5

# ---------------------------------------------------------------------------
# کلید API
# ---------------------------------------------------------------------------


def load_api_key() -> str:
    """خواندن کلید — اول از فایل api_key.txt، بعد از متغیر محیطی."""
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, encoding="utf-8") as f:
            key = f.read().strip()
            if key:
                return key
    key = os.environ.get("AVALAI_API_KEY", "").strip()
    if key:
        return key
    print(
        "خطا: کلید API یافت نشد. یکی از این دو را انجام دهید:\n"
        f"  ۱) کلید را در فایل {KEY_FILE} ذخیره کنید\n"
        "  ۲) متغیر محیطی AVALAI_API_KEY را تنظیم کنید",
        file=sys.stderr,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# لنگر موضوع: استخراج جمله اول متن انسانی برای جلوگیری از توهم
# ---------------------------------------------------------------------------


def get_topic_anchor(text: str, n_words: int = 15) -> str:
    """بازگرداندن ۱۵ کلمه اول متن انسانی به‌عنوان لنگر موضوع."""
    words = text.split()
    return " ".join(words[:n_words])


def strip_copied_anchor(generated: str, anchor: str) -> str:
    """اگر مدل لنگر را کلمه‌به‌کلمه کپی کرده، آن را از ابتدای خروجی حذف کن."""
    if not anchor:
        return generated
    norm_anchor = re.sub(r"\s+", " ", anchor).strip()
    norm_gen = re.sub(r"\s+", " ", generated).strip()
    if norm_gen.startswith(norm_anchor):
        return norm_gen[len(norm_anchor):].lstrip(" ،.؛:")
    return generated


# ---------------------------------------------------------------------------
# پرامپت
# ---------------------------------------------------------------------------


def build_prompt(human_record: dict) -> str:
    """ساخت پرامپت با لنگر موضوع و کنترل طول."""
    anchor = get_topic_anchor(human_record.get("text", ""), 15)
    target = human_record.get("num_words", 100)
    return (
        "یک متن کوتاه فارسی به سبک دانشنامه‌ای بنویس درباره موضوع زیر.\n"
        f"موضوع (جمله آغازین برای مرجع): {anchor}\n\n"
        "قواعد:\n"
        f"• طول متن حدود {target} کلمه باشد.\n"
        "• با زبان رسمی و خنثی بنویس.\n"
        "• جمله آغازین داده‌شده را عیناً تکرار نکن؛ موضوع را با کلمات خودت ادامه بده.\n"
        "• از علامت‌گذاری مارک‌داون، عنوان، یا فهرست استفاده نکن.\n"
    )


# ---------------------------------------------------------------------------
# فراخوانی مدل
# ---------------------------------------------------------------------------


def model_sleep(model: str) -> float:
    """فاصله بین درخواست‌ها بر اساس RPM مدل، با ضریب امنیتی ۲۰٪."""
    rpm = RPM.get(model, 60)
    return (60.0 / rpm) * 1.2


def call_model(model: str, prompt: str, api_key: str) -> str | None:
    """ارسال درخواست به یک مدل با تلاش مجدد در خطاهای موقت."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 600,
    }
    backoff = 5.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(
                API_BASE, headers=headers, json=payload, timeout=REQUEST_TIMEOUT
            )
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.HTTPError as err:
            print(f"      خطای HTTP ({model}، تلاش {attempt}): {err}")
            if r.status_code in (400, 401, 403, 404):
                return None  # خطای ساختاری — تلاش مجدد بی‌فایده است
            time.sleep(backoff)
            backoff *= 2
        except (requests.RequestException, KeyError, ValueError) as err:
            print(f"      خطا ({model}، تلاش {attempt}): {err}")
            time.sleep(backoff)
            backoff *= 2
    return None


# ---------------------------------------------------------------------------
# ورودی/خروجی JSONL
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> Iterable[dict]:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_done_pairs() -> set[tuple[str, str]]:
    """جفت‌های (شناسه_انسان، مدل) که قبلاً تولید شده‌اند."""
    done: set[tuple[str, str]] = set()
    for rec in load_jsonl(OUT_PATH):
        human_id = rec.get("human_id")
        if not human_id:
            # سازگاری با قالب قدیمی "ai_0001"
            raw = rec.get("id", "")
            human_id = raw[3:] if raw.startswith("ai_") else raw
            if "_" in human_id:
                human_id = human_id.split("_")[0]
        done.add((human_id, rec.get("model")))
    return done


# ---------------------------------------------------------------------------
# حلقه اصلی
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="تولید متن‌های AI موازی")
    parser.add_argument(
        "--test",
        action="store_true",
        help="حالت آزمایش با فقط ۹ متن (۳ موضوع × ۳ مدل)",
    )
    args = parser.parse_args()

    api_key = load_api_key()

    humans = list(load_jsonl(INPUT_PATH))
    if args.test:
        print("=== حالت آزمایش: فقط ۹ متن ===")
        humans = humans[:3]
    print(f"بارگذاری {len(humans)} متن انسانی.")
    print(f"مدل‌ها: {MODELS}")
    print(
        "فاصله هر مدل (ثانیه): "
        + ", ".join(f"{m}={model_sleep(m):.1f}" for m in MODELS)
    )

    done_pairs = load_done_pairs()
    if done_pairs:
        print(f"  {len(done_pairs)} نسخه قبلاً تولید شده؛ از آن‌ها رد می‌شویم.")

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    count = 0

    with open(OUT_PATH, "a", encoding="utf-8") as fout:
        for model in MODELS:
            tag = MODEL_TAG.get(model, model.replace(".", "").replace("-", ""))
            remaining = [
                h for h in humans if (h["id"], model) not in done_pairs
            ]
            if not remaining:
                print(f"\n=== مدل {model}: همه نسخه‌ها از قبل موجود است ===")
                continue
            print(f"\n=== پردازش با مدل {model}: {len(remaining)} متن ===")

            fail_streak = 0
            for h in remaining:
                ai_id = f"ai_{h['id']}_{tag}"
                anchor = get_topic_anchor(h.get("text", ""), 15)
                text = call_model(model, build_prompt(h), api_key)
                if text:
                    text = strip_copied_anchor(text, anchor)
                if not text:
                    print(f"  رد شد: {h['title']}")
                    fail_streak += 1
                    if fail_streak >= FAIL_STREAK_LIMIT:
                        print(
                            f"  ⚠️ مدل {model} ظاهراً در دسترس نیست. "
                            f"به مدل بعدی می‌رویم؛ بعداً برای این مدل دوباره اجرا کنید."
                        )
                        break
                    continue
                fail_streak = 0

                record = {
                    "id": ai_id,
                    "human_id": h["id"],
                    "category": h.get("category"),
                    "subcategory": h.get("subcategory"),
                    "title": h["title"],
                    "num_words": len(text.split()),
                    "target_words": h.get("num_words"),
                    "source": "ai",
                    "model": model,
                    "generation_date": time.strftime("%Y-%m-%d"),
                    "text": text,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                count += 1
                if count % 20 == 0:
                    print(f"  تولید شده: {count} (مدل فعلی: {model})")
                time.sleep(model_sleep(model))

    print(f"\nمجموع تولیدشده در این اجرا: {count} متن → {OUT_PATH}")
    print("گام بعدی: python src/build_dataset.py")


if __name__ == "__main__":
    main()
