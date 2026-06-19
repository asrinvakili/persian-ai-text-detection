"""
extract_wikipedia.py
====================
استخراج متن‌های انسانی فارسی از ویکی‌پدیای فارسی، با نمونه‌گیری طبقه‌بندی‌شده از پنج
حوزه و سقف هر زیردسته. همه متن‌ها از نسخه‌های پیش از ۲۰۲۰ گرفته می‌شوند تا تضمین
شود که داده انسانی واقعی است و آلوده به متن تولیدی LLMها نیست.

خروجی: data/human_texts.jsonl + data/sampling_report.json

محیط:
  • Python 3.10+
  • requests
"""

from __future__ import annotations

import json
import os
import random
import re
import time
from collections import defaultdict
from typing import Iterable

import requests

# ---------------------------------------------------------------------------
# پیکربندی
# ---------------------------------------------------------------------------

API_URL = "https://fa.wikipedia.org/w/api.php"
USER_AGENT = (
    "PersianAITextDetection/1.0 "
    "(research; https://github.com/asrinvakili/persian-ai-text-detection)"
)
CUTOFF_DATE = "2019-12-31T23:59:59Z"
SEED = 42
MAX_PER_SUBCAT = 15           # سقف متن از یک زیردسته (برای جلوگیری از سوگیری)
SUBCAT_DEPTH = 1              # عمق گردش در درخت دسته‌بندی ویکی‌پدیا
SAMPLES_PER_CATEGORY = 200    # هدف برای هر یک از ۵ حوزه
MIN_WORDS = 40                # حداقل طول متن قابل قبول
MAX_WORDS = 300               # سقف برش نهایی
SLEEP_BETWEEN_REQUESTS = 0.6  # ثانیه — برای رعایت محدودیت نرخ ویکی‌پدیا
MAX_RETRIES = 5
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "data")

CATEGORIES = {
    "علوم نظری": ["ریاضیات", "فیزیک", "شیمی", "آمار", "منطق"],
    "علوم مهندسی": [
        "مهندسی برق",
        "مهندسی کامپیوتر",
        "مهندسی مکانیک",
        "مهندسی عمران",
        "مهندسی شیمی",
    ],
    "علوم پزشکی": ["پزشکی", "زیست‌شناسی", "داروشناسی", "تشریح", "بیماری‌ها"],
    "علوم انسانی": ["فلسفه", "روان‌شناسی", "جامعه‌شناسی", "زبان‌شناسی", "ادبیات"],
    "تاریخ و زندگی‌نامه": [
        "تاریخ ایران",
        "تاریخ جهان",
        "زندگی‌نامه دانشمندان",
        "زندگی‌نامه نویسندگان",
        "باستان‌شناسی",
    ],
}

# ---------------------------------------------------------------------------
# سرویس HTTP با مدیریت ۴۲۹ و بازآزمایی نمایی
# ---------------------------------------------------------------------------

session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})


def wiki_request(params: dict) -> dict | None:
    """ارسال درخواست به API ویکی‌پدیا با تلاش مجدد در صورت خطا."""
    backoff = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(API_URL, params=params, timeout=30)
            if response.status_code == 429:
                time.sleep(backoff)
                backoff *= 2
                continue
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as err:
            if attempt == MAX_RETRIES - 1:
                print(f"      درخواست ناموفق: {err}")
                return None
            time.sleep(backoff)
            backoff *= 2
    return None


# ---------------------------------------------------------------------------
# گردش در دسته‌بندی‌ها و گردآوری مقالات نامزد
# ---------------------------------------------------------------------------


def list_pages_in_category(category: str, limit: int = 500) -> list[dict]:
    """بازگرداندن لیستی از صفحات یک دسته‌بندی ویکی‌پدیا (شامل زیردسته‌ها)."""
    params = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": f"رده:{category}",
        "cmlimit": min(limit, 500),
        "cmtype": "page|subcat",
        "format": "json",
    }
    data = wiki_request(params)
    if not data:
        return []
    return data.get("query", {}).get("categorymembers", [])


def collect_candidates(root_category: str) -> dict[str, list[str]]:
    """جمع‌آوری نامزدها از یک دسته‌ریشه و زیردسته‌های آن (تا عمق SUBCAT_DEPTH)."""
    candidates: dict[str, list[str]] = defaultdict(list)
    members = list_pages_in_category(root_category)
    for m in members:
        ns = m.get("ns")
        title = m.get("title", "")
        if ns == 0:
            candidates[root_category].append(title)
        elif ns == 14 and SUBCAT_DEPTH >= 1:
            sub_name = title.replace("رده:", "").strip()
            sub_members = list_pages_in_category(sub_name)
            for sm in sub_members:
                if sm.get("ns") == 0:
                    candidates[sub_name].append(sm.get("title", ""))
            time.sleep(SLEEP_BETWEEN_REQUESTS)
        time.sleep(SLEEP_BETWEEN_REQUESTS / 3)
    return candidates


# ---------------------------------------------------------------------------
# واکشی نسخه پیش از ۲۰۲۰
# ---------------------------------------------------------------------------


def get_pre_cutoff_revision(title: str) -> int | None:
    """بازگرداندن شناسه نسخه‌ای از مقاله که پیش از CUTOFF_DATE ثبت شده باشد."""
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": title,
        "rvprop": "ids|timestamp",
        "rvlimit": 1,
        "rvstart": CUTOFF_DATE,
        "rvdir": "older",
        "format": "json",
    }
    data = wiki_request(params)
    if not data:
        return None
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        revs = page.get("revisions", [])
        if revs:
            return revs[0].get("revid")
    return None


def fetch_revision_extract(revid: int) -> str | None:
    """بازگرداندن متن خلاصه‌شده یک نسخه (بخش اول مقاله)."""
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": "1",
        "exintro": "1",
        "revids": revid,
        "format": "json",
    }
    data = wiki_request(params)
    if not data:
        return None
    pages = data.get("query", {}).get("pages", {})
    for _, page in pages.items():
        text = page.get("extract")
        if text:
            return text
    return None


# ---------------------------------------------------------------------------
# پاکسازی متن
# ---------------------------------------------------------------------------

CLEANUP_PATTERNS = [
    re.compile(r"\[\d+\]"),                # ارجاع‌های شماره‌دار
    re.compile(r"\([^)]*؟[^)]*\)"),       # پرانتزهای پرسشی
    re.compile(r"\s+"),                    # فضای خالی تکراری
]


def clean_text(text: str) -> str:
    text = text.strip()
    for pat in CLEANUP_PATTERNS[:2]:
        text = pat.sub("", text)
    text = CLEANUP_PATTERNS[2].sub(" ", text)
    return text.strip()


def trim_to_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


# ---------------------------------------------------------------------------
# منطق نمونه‌گیری
# ---------------------------------------------------------------------------


def stratified_sample(
    candidates: dict[str, list[str]],
    target: int,
    rng: random.Random,
) -> list[tuple[str, str]]:
    """انتخاب متعادل از زیردسته‌ها با سقف MAX_PER_SUBCAT."""
    selected: list[tuple[str, str]] = []
    subcats = list(candidates.keys())
    rng.shuffle(subcats)
    quotas = {sc: min(MAX_PER_SUBCAT, len(candidates[sc])) for sc in subcats}
    pools = {sc: list(candidates[sc]) for sc in subcats}
    for sc in subcats:
        rng.shuffle(pools[sc])

    while len(selected) < target:
        added_this_round = False
        for sc in subcats:
            if quotas[sc] > 0 and pools[sc]:
                title = pools[sc].pop()
                selected.append((sc, title))
                quotas[sc] -= 1
                added_this_round = True
                if len(selected) >= target:
                    break
        if not added_this_round:
            break
    return selected


# ---------------------------------------------------------------------------
# نقطه ورود
# ---------------------------------------------------------------------------


def main() -> None:
    rng = random.Random(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "human_texts.jsonl")
    report_path = os.path.join(OUTPUT_DIR, "sampling_report.json")

    report: dict[str, dict] = {}
    written = 0

    with open(out_path, "w", encoding="utf-8") as fout:
        for category, root_subcats in CATEGORIES.items():
            print(f"\n=== حوزه: {category} ===")
            all_candidates: dict[str, list[str]] = defaultdict(list)
            for root in root_subcats:
                got = collect_candidates(root)
                for sc, titles in got.items():
                    all_candidates[sc].extend(titles)
            for sc in all_candidates:
                all_candidates[sc] = list(set(all_candidates[sc]))
            total_cand = sum(len(v) for v in all_candidates.values())
            print(f"  نامزدهای جمع‌شده: {total_cand} از {len(all_candidates)} زیردسته")

            picks = stratified_sample(all_candidates, SAMPLES_PER_CATEGORY, rng)
            cat_subcats: set[str] = set()
            cat_word_counts: list[int] = []
            kept = 0

            for subcat, title in picks:
                revid = get_pre_cutoff_revision(title)
                time.sleep(SLEEP_BETWEEN_REQUESTS)
                if not revid:
                    continue
                raw = fetch_revision_extract(revid)
                time.sleep(SLEEP_BETWEEN_REQUESTS)
                if not raw:
                    continue
                text = clean_text(raw)
                wc = len(text.split())
                if wc < MIN_WORDS:
                    continue
                text = trim_to_words(text, MAX_WORDS)
                wc = len(text.split())
                record = {
                    "id": f"{written:04d}",
                    "category": category,
                    "subcategory": subcat,
                    "title": title,
                    "revid": revid,
                    "num_words": wc,
                    "source": "human",
                    "text": text,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()
                written += 1
                kept += 1
                cat_subcats.add(subcat)
                cat_word_counts.append(wc)
                if kept >= SAMPLES_PER_CATEGORY:
                    break

            mean_wc = sum(cat_word_counts) / max(len(cat_word_counts), 1)
            stdev_wc = (
                (sum((x - mean_wc) ** 2 for x in cat_word_counts) / max(len(cat_word_counts), 1))
                ** 0.5
            )
            report[category] = {
                "kept": kept,
                "distinct_subcategories": len(cat_subcats),
                "mean_words": round(mean_wc, 1),
                "stdev_words": round(stdev_wc, 1),
            }
            print(
                f"  حفظ شد: {kept} | زیردسته‌های متمایز: {len(cat_subcats)} "
                f"| میانگین کلمات: {mean_wc:.1f}"
            )

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\nمجموع متن‌های انسانی نوشته‌شده: {written} → {out_path}")
    print(f"گزارش نمونه‌گیری: {report_path}")


if __name__ == "__main__":
    main()
