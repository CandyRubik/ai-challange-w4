"""Render a silent 36-second MP4 summary from the recorded Day 22 evaluation.

Usage: python -m evaluation.make_day22_video
Requires rsvg-convert and ffmpeg; reads data/day22-evaluation.json.
"""

from __future__ import annotations

import argparse
from html import escape
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory


DEFAULT_REPORT = Path("data/day22-evaluation.json")
DEFAULT_OUTPUT = Path("artifacts/day22-rag-comparison.mp4")
REVIEW_PATH = Path(__file__).with_name("day22-review.json")
WHITE = "#f4f2ff"
MUTED = "#a9aac1"
PURPLE = "#a695ff"
GREEN = "#73e6b8"
RED = "#ff8fa3"


def lines(x: int, y: int, items: list[str], *, size: int = 27,
          color: str = WHITE, weight: int = 400, gap: int = 41) -> str:
    return "".join(
        f'<text x="{x}" y="{y + i * gap}" font-family="Arial" font-size="{size}" '
        f'font-weight="{weight}" fill="{color}">{escape(item)}</text>'
        for i, item in enumerate(items)
    )


def box(x: int, y: int, width: int, height: int, *, fill: str = "#1b1c31",
        stroke: str = "#393858") -> str:
    return (f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="22" fill="{fill}" stroke="{stroke}" stroke-width="2"/>')


def frame(number: int, title: str, content: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
    <defs><linearGradient id="bg" x2="1" y2="1"><stop stop-color="#11132b"/>
    <stop offset="1" stop-color="#080a15"/></linearGradient></defs>
    <rect width="1280" height="720" fill="url(#bg)"/>
    <circle cx="1110" cy="100" r="210" fill="#5946bb" opacity=".09"/>
    <rect x="65" y="61" width="74" height="6" rx="3" fill="{PURPLE}"/>
    {lines(65, 112, ["AI CHALLENGE W4  ·  ДЕНЬ 22"], size=18, color=PURPLE, weight=700)}
    {lines(65, 174, [title], size=42, weight=700)}
    {content}
    <line x1="65" y1="666" x2="1215" y2="666" stroke="#393858"/>
    {lines(65, 695, ["Java Concurrency in Practice  ·  10 контрольных вопросов"], size=18, color=MUTED)}
    {lines(1180, 695, [f"{number}/6"], size=18, color=MUTED)}
    </svg>'''


def slides(plain: int, rag: int, retrieved: int, cited: int) -> list[str]:
    first = frame(1, "Первый RAG-запрос", "".join([
        lines(66, 252, ["Один вопрос  ·  два режима  ·  одна модель"], size=29, color=MUTED),
        box(65, 310, 1150, 250, fill="#211e3d", stroke="#6a5bc6"),
        lines(104, 381, ["БЕЗ RAG", ""], size=22, color=MUTED, weight=700),
        lines(103, 489, [f"{plain}/10"], size=82, color=WHITE, weight=700),
        lines(584, 489, ["→"], size=82, color=PURPLE, weight=700),
        lines(765, 381, ["С RAG"], size=22, color=GREEN, weight=700),
        lines(765, 489, [f"{rag}/10"], size=82, color=GREEN, weight=700),
    ]))

    second = frame(2, "Как работают два режима", "".join([
        lines(65, 245, ["БЕЗ RAG"], size=22, color=MUTED, weight=700),
        box(65, 270, 1150, 112),
        lines(94, 339, ["Вопрос    →    DeepSeek    →    ответ"], size=33),
        lines(65, 447, ["С RAG"], size=22, color=GREEN, weight=700),
        box(65, 472, 1150, 145, fill="#162c2b", stroke="#347c66"),
        lines(94, 528, ["Вопрос  →  поиск top-3  →  фрагменты + вопрос"], size=31),
        lines(94, 578, ["→  DeepSeek  →  ответ с проверяемыми ссылками"], size=31),
    ]))

    third = frame(3, "Где поиск помог", "".join([
        box(65, 225, 552, 372), box(640, 225, 575, 372, fill="#17302c", stroke="#347c66"),
        lines(95, 273, ["БЕЗ RAG"], size=21, color=MUTED, weight=700),
        lines(668, 273, ["С RAG"], size=21, color=GREEN, weight=700),
        lines(95, 328, ["Вопрос 3: execution policy", "Ответ про PowerShell", "вместо политики выполнения", "задач из книги."], size=24, gap=42),
        lines(668, 328, ["Шесть решений политики", "из §6.2.2, PDF с. 7.", "Вопрос 9: timed invokeAll,", "таймаут и запасной результат."], size=24, gap=42),
        lines(95, 558, ["✕  Контекст не угадан"], size=23, color=RED, weight=700),
        lines(668, 558, ["✓  Источник найден"], size=23, color=GREEN, weight=700),
    ]))

    fourth = frame(4, "Где RAG ошибся", "".join([
        box(65, 230, 1150, 352, fill="#30202c", stroke="#8c4a62"),
        lines(96, 287, ["Вопрос 8: CompletionService"], size=29, color=RED, weight=700),
        lines(96, 345, ["Нужный чанк найден и процитирован."], size=27),
        lines(96, 399, ["Но модель написала, что poll() блокируется,"], size=27),
        lines(96, 440, ["когда очередь пуста."], size=27),
        lines(96, 508, ["Обычный poll() возвращает null сразу; ждёт take()."], size=27, color=GREEN),
        lines(96, 551, ["Цитата сама по себе не гарантирует точный ответ."], size=23, color=MUTED),
    ]))

    fifth = frame(5, "Итог проверки", "".join([
        box(65, 231, 1150, 348),
        lines(99, 290, ["Полные и точные ответы"], size=26, color=MUTED),
        lines(874, 290, [f"{plain}/10  →  {rag}/10"], size=30, color=GREEN, weight=700),
        lines(99, 362, ["Нужный источник в top-3"], size=26, color=MUTED),
        lines(1000, 362, [f"{retrieved}/9"], size=30, color=GREEN, weight=700),
        lines(99, 434, ["Подтверждённые ответы с цитатами"], size=26, color=MUTED),
        lines(1000, 434, [f"{cited}/9"], size=30, color=GREEN, weight=700),
        lines(99, 506, ["Kotlin: данных нет, ложных ссылок нет"], size=26, color=MUTED),
        lines(1000, 506, ["✓"], size=30, color=GREEN, weight=700),
    ]))

    sixth = frame(6, "Повторить результат", "".join([
        box(65, 241, 1150, 308, fill="#211e3d", stroke="#6a5bc6"),
        lines(97, 298, ["Код: два режима + 10 вопросов + парный прогон"], size=28),
        lines(97, 362, ["python -m evaluation.run_day22"], size=30, color=PURPLE, weight=700),
        lines(97, 419, ["Отчёт: docs/day22-results.md"], size=27),
        lines(97, 464, ["Полные ответы: data/day22-evaluation.json"], size=27),
        lines(97, 517, ["Один прогон; качество цитат проверено отдельно."], size=23, color=MUTED),
    ]))
    return [first, second, third, fourth, fifth, sixth]


def make_video(report_path: Path, output: Path) -> None:
    if not shutil.which("rsvg-convert") or not shutil.which("ffmpeg"):
        raise RuntimeError("Для видео нужны rsvg-convert и ffmpeg")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    review_data = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    if review_data["report_sha256"] != hashlib.sha256(report_path.read_bytes()).hexdigest():
        raise ValueError("Результаты изменились; сначала обновите ручную оценку")
    review = review_data["cases"]
    if [row["id"] for row in review] != [row["id"] for row in report["cases"]]:
        raise ValueError("Ручная оценка не соответствует контрольным вопросам")
    plain = sum(row["plain_pass"] for row in review)
    rag = sum(row["rag_pass"] for row in review)
    retrieved = sum(case["answers"]["rag"]["expected_source_in_top_3"] is True
                    for case in report["cases"])
    cited = sum(row["citation_supported"] is True for row in review)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="day22-video-") as temporary_directory:
        directory = Path(temporary_directory)
        for i, svg in enumerate(slides(plain, rag, retrieved, cited)):
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    make_video(args.report, args.output)
    print(f"Saved video to {args.output}")


if __name__ == "__main__":
    main()
