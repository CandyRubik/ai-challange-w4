"""Render a silent Day 23 comparison video from the evaluated report and review.

Usage: python -m evaluation.make_day23_video
Requires rsvg-convert and ffmpeg.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

from .make_day22_video import GREEN, MUTED, PURPLE, RED, WHITE, box, lines


DEFAULT_REPORT = Path("data/day23-evaluation.json")
DEFAULT_REVIEW = Path(__file__).with_name("day23-review.json")
DEFAULT_OUTPUT = Path("artifacts/day23-rag-reranking.mp4")


def frame(number: int, title: str, content: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
    <defs><linearGradient id="bg" x2="1" y2="1"><stop stop-color="#11132b"/>
    <stop offset="1" stop-color="#080a15"/></linearGradient></defs>
    <rect width="1280" height="720" fill="url(#bg)"/>
    <circle cx="1110" cy="100" r="210" fill="#5946bb" opacity=".09"/>
    <rect x="65" y="61" width="74" height="6" rx="3" fill="{PURPLE}"/>
    {lines(65, 112, ["AI CHALLENGE W4  ·  ДЕНЬ 23"], size=18, color=PURPLE, weight=700)}
    {lines(65, 174, [title], size=42, weight=700)}
    {content}
    <line x1="65" y1="666" x2="1215" y2="666" stroke="#393858"/>
    {lines(65, 695, ["Java Concurrency in Practice  ·  контрольный прогон"], size=18, color=MUTED)}
    {lines(1180, 695, [f"{number}/6"], size=18, color=MUTED)}
    </svg>'''


def slides(report: dict, review: dict) -> list[str]:
    cases = report["cases"]
    positives = sum(bool(case["sections"]) for case in cases)
    negatives = len(cases) - positives
    metrics = {
        name: (
            sum(case["modes"][name]["expected_in_selected"] is True for case in cases),
            sum(case["modes"][name]["empty_for_negative"] is True for case in cases),
        )
        for name in ("raw", "rewrite", "filter", "enhanced")
    }
    reviews = review["cases"]
    raw_facts = sum(row["raw_facts"] for row in reviews)
    enhanced_facts = sum(row["enhanced_facts"] for row in reviews)
    raw_citations = sum(row["raw_citations"] for row in reviews)
    enhanced_citations = sum(row["enhanced_citations"] for row in reviews)
    example = next(case for case in cases if case["id"] == 10)

    return [
        frame(1, "Реранкинг и фильтрация", "".join([
            lines(65, 263, ["Вопрос  →  query rewrite  →  FAISS top-10"], size=32),
            box(65, 314, 1150, 210, fill="#211e3d", stroke="#6a5bc6"),
            lines(98, 385, ["Исходный вопрос + 10 кандидатов  →  reranker"], size=32),
            lines(98, 452, ["Порог релевантности  →  до 3 фрагментов  →  ответ"], size=31, color=GREEN),
            lines(65, 586, ["Если подходящих фрагментов нет — ответ без ссылок."], size=26, color=MUTED),
        ])),
        frame(2, "Почему одного similarity мало", "".join([
            box(65, 236, 1150, 350, fill="#30202c", stroke="#8c4a62"),
            lines(99, 298, ["Вопрос: синтаксис Kotlin coroutines в книге?"], size=30),
            lines(99, 365, ["Прежний rewrite потерял Kotlin и coroutines."], size=27),
            lines(99, 425, ["Нерелевантный результат получил score 0.903."], size=27, color=RED),
            lines(99, 498, ["Теперь термины сохраняются, кандидаты проверяются", "по исходному вопросу и техническим именам."], size=26, color=GREEN),
        ])),
        frame(3, "Четыре режима поиска", "".join([
            lines(75, 245, ["Режим", "raw", "rewrite", "filter", "enhanced"], size=26, gap=72),
            lines(565, 245, ["Нужный источник", *[
                f"{metrics[name][0]}/{positives}" for name in metrics
            ]], size=26, gap=72),
            lines(940, 245, ["Пусто на отриц.", *[
                f"{metrics[name][1]}/{negatives}" for name in metrics
            ]], size=26, gap=72),
            lines(75, 608, [f"top-{report['top_k_before']} до фильтра  ·  top-{report['top_k_after']} после  ·  порог {report['threshold']}"], size=23, color=MUTED),
        ])),
        frame(4, "Проверка ответов", "".join([
            box(65, 231, 1150, 337),
            lines(99, 295, ["Критерий", "Точные факты", "Корректные ссылки / их отсутствие"], size=25, gap=86),
            lines(813, 295, ["raw", f"{raw_facts}/{len(cases)}", f"{raw_citations}/{len(cases)}"], size=27, gap=86),
            lines(1010, 295, ["enhanced", f"{enhanced_facts}/{len(cases)}", f"{enhanced_citations}/{len(cases)}"], size=27, color=GREEN, gap=86),
            lines(99, 527, ["Факты и ссылки оценены отдельно вручную."], size=22, color=MUTED),
        ])),
        frame(5, "Отрицательный контроль", "".join([
            box(65, 236, 1150, 330, fill="#17302c", stroke="#347c66"),
            lines(99, 295, ["Kotlin coroutines: подтверждения в sample PDF нет."], size=29),
            lines(99, 370, [f"raw: {len(example['modes']['raw']['selected_ids'])} фрагмента", f"enhanced: {len(example['modes']['enhanced']['selected_ids'])} фрагментов"], size=29, gap=60),
            lines(99, 527, ["Результат фильтра: никакой случайной ссылки на книгу."], size=26, color=GREEN),
        ])),
        frame(6, "Код и воспроизведение", "".join([
            box(65, 240, 1150, 310, fill="#211e3d", stroke="#6a5bc6"),
            lines(96, 301, ["Реализация: app/indexing/rag.py + rerank.py"], size=27),
            lines(96, 366, ["python -m evaluation.run_day23"], size=30, color=PURPLE, weight=700),
            lines(96, 423, ["Отчёт и разбор: docs/day23-results.md"], size=27),
            lines(96, 487, ["Один прогон на 15 вопросах; модель может отвечать иначе."], size=23, color=MUTED),
        ])),
    ]


def make_video(report_path: Path, review_path: Path, output: Path) -> None:
    if not shutil.which("rsvg-convert") or not shutil.which("ffmpeg"):
        raise RuntimeError("Для видео нужны rsvg-convert и ffmpeg")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if report["retrieval_only"]:
        raise ValueError("Для видео нужен полный прогон с ответами")
    if review["report_sha256"] != hashlib.sha256(report_path.read_bytes()).hexdigest():
        raise ValueError("Ручная проверка относится к другой версии отчёта")
    if [row["id"] for row in review["cases"]] != [case["id"] for case in report["cases"]]:
        raise ValueError("Ручная проверка не соответствует вопросам")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="day23-video-") as temporary_directory:
        directory = Path(temporary_directory)
        for i, svg in enumerate(slides(report, review)):
            source = directory / f"slide-{i:02d}.svg"
            source.write_text(svg, encoding="utf-8")
            subprocess.run([
                "rsvg-convert", "--width", "1280", "--height", "720",
                "--output", str(directory / f"slide-{i:02d}.png"), str(source),
            ], check=True)
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-framerate", "1/6", "-i", str(directory / "slide-%02d.png"),
            "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-crf", "20",
            "-movflags", "+faststart", "-t", "36", str(output),
        ], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    make_video(args.report, args.review, args.output)
    print(f"Saved video to {args.output}")


if __name__ == "__main__":
    main()
