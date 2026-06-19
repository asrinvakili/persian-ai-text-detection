"""
build_dataset.py
================
ساخت دیتاست نهایی از فایل‌های human_texts.jsonl و ai_texts.jsonl، تقسیم به
آموزش/اعتبارسنجی/آزمون با لایه‌بندی بر اساس (برچسب، دسته، مدل).

ویژگی طرح تطبیقی: هر موضوع انسانی نسخه‌ای از هر مدل دارد، پس نسبت کلاس
انسان به AI طبیعتاً ۱:۳ است. این عدم تعادل در هنگام آموزش مدل تشخیص با
یکی از این روش‌ها مدیریت می‌شود:
  • وزن‌دهی کلاس (class_weight) — همه داده حفظ می‌شود.
  • زیرنمونه‌گیری — از هر موضوع یک نسخه AI تصادفی بردارید.

خروجی:
  data/train.jsonl
  data/val.jsonl
  data/test.jsonl
  data/dataset_stats.json
"""

from __future__ import annotations

import json
import os
import random
import statistics
from collections import defaultdict
from typing import Iterable

# ---------------------------------------------------------------------------
# پیکربندی
# ---------------------------------------------------------------------------

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "data")
HUMAN_PATH = os.path.join(OUTPUT_DIR, "human_texts.jsonl")
AI_PATH = os.path.join(OUTPUT_DIR, "ai_texts.jsonl")

TEST_FRACTION = 0.20
VAL_FRACTION = 0.10
SEED = 42

# ---------------------------------------------------------------------------
# ابزارها
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> Iterable[dict]:
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(records: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# تقسیم لایه‌بندی‌شده با آگاهی از مدل
# ---------------------------------------------------------------------------


def stratified_split(
    records: list[dict],
    test_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    تقسیم لایه‌بندی‌شده بر اساس (برچسب، دسته، مدل).
    افزودن مدل به کلید لایه‌بندی تضمین می‌کند هر بخش (آموزش/آزمون) از هر
    مدل سهم متناسب بگیرد — برای تحلیل «کدام مدل سخت‌تر تشخیص داده می‌شود»
    ضروری است.
    """
    rng = random.Random(seed)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        model = r.get("model", "human") if r["label"] == 1 else "human"
        key = (r["label"], r.get("category", "نامشخص"), model)
        groups[key].append(r)

    train, val, test = [], [], []
    for key, items in groups.items():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, int(n * test_frac)) if n >= 5 else 0
        n_val = max(1, int(n * val_frac)) if n >= 10 else 0
        test.extend(items[:n_test])
        val.extend(items[n_test : n_test + n_val])
        train.extend(items[n_test + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


# ---------------------------------------------------------------------------
# آمار
# ---------------------------------------------------------------------------


def compute_stats(records: list[dict]) -> dict:
    by_label = defaultdict(int)
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_model: dict[str, int] = defaultdict(int)
    word_counts: dict[str, list[int]] = {"human": [], "AI": []}

    for r in records:
        lab = "human" if r["label"] == 0 else "AI"
        by_label[lab] += 1
        by_category[r.get("category", "نامشخص")][lab] += 1
        wc = r.get("num_words", 0)
        word_counts[lab].append(wc)
        if r["label"] == 1:
            by_model[r.get("model", "نامشخص")] += 1

    def summarize(values: list[int]) -> dict:
        if not values:
            return {}
        return {
            "mean": round(statistics.mean(values), 1),
            "stdev": round(statistics.stdev(values), 1) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
        }

    return {
        "total": len(records),
        "by_label": dict(by_label),
        "by_category": {k: dict(v) for k, v in by_category.items()},
        "ai_by_model": dict(by_model),
        "word_count": {k: summarize(v) for k, v in word_counts.items()},
    }


# ---------------------------------------------------------------------------
# نقطه ورود
# ---------------------------------------------------------------------------


def main() -> None:
    if not os.path.exists(HUMAN_PATH):
        print(f"خطا: {HUMAN_PATH} یافت نشد. اول extract_wikipedia.py را اجرا کنید.")
        return
    if not os.path.exists(AI_PATH):
        print(f"خطا: {AI_PATH} یافت نشد. اول generate_ai.py را اجرا کنید.")
        return

    records: list[dict] = []
    for rec in load_jsonl(HUMAN_PATH):
        rec["label"] = 0
        records.append(rec)
    for rec in load_jsonl(AI_PATH):
        rec["label"] = 1
        records.append(rec)

    train, val, test = stratified_split(records, TEST_FRACTION, VAL_FRACTION, SEED)

    write_jsonl(train, os.path.join(OUTPUT_DIR, "train.jsonl"))
    write_jsonl(val, os.path.join(OUTPUT_DIR, "val.jsonl"))
    write_jsonl(test, os.path.join(OUTPUT_DIR, "test.jsonl"))

    stats = compute_stats(records)
    stats["splits"] = {"train": len(train), "val": len(val), "test": len(test)}
    with open(os.path.join(OUTPUT_DIR, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    h = stats["by_label"].get("human", 0)
    a = stats["by_label"].get("AI", 0)
    print(f"دیتاست نهایی: {len(records)} نمونه")
    print(f"  انسان: {h} | AI: {a}")
    print(f"  تقسیم: آموزش={len(train)} | اعتبارسنجی={len(val)} | آزمون={len(test)}")
    print(f"آمار کامل در: {os.path.join(OUTPUT_DIR, 'dataset_stats.json')}")

    if h and a:
        ratio = max(h, a) / min(h, a)
        if ratio > 1.5:
            print(
                f"⚠️ عدم تعادل کلاس: {h} انسان در برابر {a} AI (نسبت {ratio:.1f})."
            )
            print("   این طبیعی است چون هر موضوع چند نسخه AI دارد (طرح تطبیقی).")
            print("   راه‌حل هنگام آموزش مدل تشخیص:")
            print("   • گزینه ۱: استفاده از وزن‌دهی کلاس (class_weight) — همه داده حفظ می‌شود.")
            print("   • گزینه ۲: برای آموزش متوازن، از هر موضوع یک نسخه AI تصادفی بردارید")
            print("     و بقیه را فقط برای تحلیل مقایسه مدل‌ها نگه دارید.")


if __name__ == "__main__":
    main()
