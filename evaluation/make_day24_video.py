"""Render a silent Day 24 RAG evidence video from the recorded run and review.

Usage: python -m evaluation.make_day24_video
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
from textwrap import wrap

from .make_day22_video import GREEN, MUTED, PURPLE, RED, WHITE, box, lines


DEFAULT_REPORT = Path(__file__).with_name("day24-evaluation.json")
DEFAULT_REVIEW = Path(__file__).with_name("day24-review.json")
DEFAULT_OUTPUT = Path("artifacts/day24-rag-citations.mp4")


def frame(number: int, title: str, content: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
    <defs><linearGradient id="bg" x2="1" y2="1"><stop stop-color="#11132b"/>
    <stop offset="1" stop-color="#080a15"/></linearGradient></defs>
    <rect width="1280" height="720" fill="url(#bg)"/>
    <circle cx="1100" cy="100" r="210" fill="#5946bb" opacity=".09"/>
    <rect x="65" y="61" width="74" height="6" rx="3" fill="{PURPLE}"/>
    {lines(65, 112, ["AI CHALLENGE W4  ·  ДЕНЬ 24"], size=18, color=PURPLE, weight=700)}
    {lines(65, 174, [title], size=42, weight=700)}
    {content}
    <line x1="65" y1="666" x2="1215" y2="666" stroke="#393858"/>
    {lines(65, 695, ["Java Concurrency in Practice  ·  10 новых вопросов"], size=18, color=MUTED)}
    {lines(1180, 695, [f"{number}/6"], size=18, color=MUTED)}
    </svg>'''


def slides(report: dict, review: dict) -> list[str]:
    cases = report["cases"]
    reviews = review["cases"]
    positives = [case for case in cases if case["sections"]]
    negatives = [case for case in cases if not case["sections"]]
    grounded = sum(case["grounding_status"] == "answered" for case in positives)
    abstained = sum(case["grounding_status"] == "insufficient_context" for case in negatives)
    automatic = sum(case["automatic_pass"] for case in cases)
    supported = sum(row["semantic_support"] is True for row in reviews)
    complete = sum(row["complete"] is True for row in reviews)
    example = cases[0]
    quote = example["citations"][0]["quote"]
    negative = negatives[1]

    return [
        frame(1, "Ответы с доказательствами", "".join([
            box(65, 236, 1150, 313, fill="#211e3d", stroke="#6a5bc6"),
            lines(97, 303, ["Вопрос  →  поиск  →  reranker  →  порог"], size=31),
            lines(97, 380, ["Тезисы + точные цитаты  →  проверка по чанкам"], size=30, color=GREEN),
            lines(97, 460, ["Сервер сам формирует ответ и список источников."], size=28),
            lines(65, 601, ["Слабый контекст: «Не знаю» и просьба уточнить."], size=26, color=MUTED),
        ])),
        frame(2, "Источник и дословная цитата", "".join([
            box(65, 226, 1150, 374, fill="#17302c", stroke="#347c66"),
            lines(96, 278, ["Вопрос 1: ThreadPerTaskWebServer"], size=26, color=GREEN, weight=700),
            lines(96, 336, wrap(example["answer"].splitlines()[0].split(" [DOC:")[0], 67),
                  size=25, gap=36),
            lines(96, 445, ["Цитата из раздела 6.1.2 · structure-0007:"], size=23, color=MUTED),
            lines(96, 483, wrap(quote, 75), size=23, gap=34),
        ])),
        frame(3, "Когда подтверждения нет", "".join([
            box(65, 240, 1150, 290, fill="#30202c", stroke="#8c4a62"),
            lines(97, 303, [negative["question"]], size=27),
            lines(97, 389, [negative["answer"]], size=29, color=RED),
            lines(97, 470, ["Источники: 0     Цитаты: 0"], size=27, color=MUTED),
            lines(65, 599, ["Нерелевантный фрагмент не выдаётся за доказательство."], size=25),
        ])),
        frame(4, "Проверка на 10 вопросах", "".join([
            box(65, 225, 1150, 368),
            lines(100, 292, ["Ответы с источниками и цитатами", "Отказ на вопросах без данных", "Структурная проверка"], size=27, gap=102),
            lines(1010, 292, [f"{grounded}/{len(positives)}", f"{abstained}/{len(negatives)}", f"{automatic}/10"],
                  size=37, color=GREEN, weight=700, gap=102),
            lines(100, 561, [f"Порог reranker: {report['threshold']} · одна новая выборка"],
                  size=22, color=MUTED),
        ])),
        frame(5, "Смысл проверен отдельно", "".join([
            box(65, 231, 1150, 348),
            lines(98, 298, ["Тезисы подтверждены цитатами", "Полные ответы по ожидаемым пунктам"],
                  size=26, gap=95),
            lines(1010, 298, [f"{supported}/{len(positives)}", f"{complete}/{len(positives)}"],
                  size=36, color=GREEN, weight=700, gap=95),
            lines(98, 513, ["Вопрос 3: смысл верен, но пропущен Future.get."],
                  size=25, color=RED),
            lines(65, 622, ["Дословная цитата сама по себе не доказывает смысл вывода."],
                  size=24, color=MUTED),
        ])),
        frame(6, "Код и воспроизведение", "".join([
            box(65, 239, 1150, 312, fill="#211e3d", stroke="#6a5bc6"),
            lines(96, 303, ["Проверка цитат: app/indexing/grounding.py"], size=26),
            lines(96, 369, ["python -m evaluation.run_day24"], size=30, color=PURPLE, weight=700),
            lines(96, 428, ["Отчёт и ручная оценка: evaluation/day24-*"], size=25),
            lines(96, 487, ["Воспроизводимость зависит от ответов модели."],
                  size=23, color=MUTED),
        ])),
    ]


def make_video(report_path: Path, review_path: Path, output: Path) -> None:
    if not shutil.which("rsvg-convert") or not shutil.which("ffmpeg"):
        raise RuntimeError("Для видео нужны rsvg-convert и ffmpeg")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review["report_sha256"] != hashlib.sha256(report_path.read_bytes()).hexdigest():
        raise ValueError("Ручная проверка относится к другой версии отчёта")
    if [row["id"] for row in review["cases"]] != [case["id"] for case in report["cases"]]:
        raise ValueError("Ручная проверка не соответствует вопросам")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="day24-video-") as temporary_directory:
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
