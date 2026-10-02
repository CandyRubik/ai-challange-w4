"""Render a silent, UI-style Day 25 video from the recorded offline run.

Usage: python -m evaluation.make_day25_video
Requires rsvg-convert and ffmpeg. No external API calls are made.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
from textwrap import wrap
from xml.sax.saxutils import escape


DEFAULT_REPORT = Path(__file__).with_name("day25-offline-evaluation.json")
DEFAULT_OUTPUT = Path("artifacts/day25-rag-memory-chat.mp4")
GREEN = "#83d8bd"
WHITE = "#ecf1f7"
MUTED = "#9bb0bd"


def box(x: int, y: int, w: int, h: int, fill: str, *, stroke: str = "none") -> str:
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" '
            f'fill="{fill}" stroke="{stroke}"/>')


def lines(x: int, y: int, content: str, *, width: int = 70, size: int = 22,
          color: str = WHITE, gap: int = 31, limit: int = 99) -> str:
    rows = wrap(content.replace("\n", " "), width=width,
                break_long_words=False, break_on_hyphens=False)
    if len(rows) > limit:
        rows = rows[:limit]
        rows[-1] = rows[-1].rstrip(". ") + "…"
    return "".join(
        f'<text x="{x}" y="{y + n * gap}" fill="{color}" '
        f'font-family="sans-serif" font-size="{size}">{escape(row)}</text>'
        for n, row in enumerate(rows)
    )


def slide(number: int, title: str, subtitle: str, detail: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="900">
    {box(0, 0, 1440, 900, "#0c121d")}
    <circle cx="1200" cy="100" r="320" fill="#173b35" opacity=".45"/>
    {box(100, 115, 75, 8, GREEN)}
    {lines(100, 210, "AI CHALLENGE W4 · ДЕНЬ 25", size=27, color=GREEN)}
    {lines(100, 365, title, size=61, gap=72)}
    {lines(100, 450, subtitle, size=31, color="#b9d4cc")}
    {box(100, 540, 1240, 150, "#182b35", stroke="#39695d")}
    {lines(135, 625, detail, size=29)}
    {lines(100, 830, "Офлайн-демо: сценарная модель · локальный индекс · проверенные источники", size=21, color=MUTED)}
    {lines(1290, 830, f"{number}/4", size=21, color=MUTED)}
    </svg>'''


def chat_slide(scenario: dict) -> str:
    turn = scenario["turns"][-1]["turn"]
    state = scenario["final_state"]
    source = next(item for item in turn["sources"] if item["kind"] == "document")
    constraint = state["constraints"][0]
    term = state["terms"][0]
    title = scenario["title"]
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="900">
    {box(0, 0, 1440, 900, "#0c121d")}
    {box(0, 0, 260, 900, "#111a27")}
    {box(1110, 0, 330, 900, "#111a27")}
    {lines(25, 50, "← ОСНОВНОЙ ЧАТ", size=15, color=MUTED)}
    {lines(25, 110, "✦  RAG-чат", size=26, color=GREEN)}
    {box(16, 162, 228, 55, "#1d4239", stroke="#3c806b")}
    {lines(28, 197, "＋ Новый RAG-чат", size=18)}
    {lines(27, 270, "ДИАЛОГИ", size=16, color=GREEN)}
    {box(16, 288, 228, 92, "#19382f", stroke="#3c806b")}
    {lines(28, 319, title, width=22, size=17, gap=24, limit=2)}
    {lines(27, 756, "БАЗА ЗНАНИЙ", size=16, color=GREEN)}
    {lines(27, 788, "Java Concurrency in Practice", width=27, size=16)}
    {lines(27, 850, "Глава 6 · индекс готов", size=15, color=MUTED)}

    {lines(295, 52, "ДИАЛОГ С ИСТОЧНИКАМИ", size=15, color=GREEN)}
    {lines(295, 92, title, width=45, size=28, limit=2)}
    {box(321, 155, 750, 117, "#1c3a36", stroke="#376b5c")}
    {lines(345, 184, "ВЫ · 12", size=14, color=GREEN)}
    {lines(345, 217, turn["content"], width=67, size=20, gap=27, limit=2)}
    {box(295, 298, 776, 421, "#182536", stroke="#304354")}
    {lines(319, 331, "RAG-ЧАТ · ОФЛАЙН-ПРОГОН", size=14, color=GREEN)}
    {lines(319, 368, turn["answer"], width=73, size=18, gap=26, limit=5)}
    {box(319, 500, 728, 1, "#314454")}
    {lines(319, 532, "ИСТОЧНИКИ", size=15, color="#b8d9ca")}
    {lines(319, 563, source["section"] + " · стр. " + str(source["page_start"]), width=67, size=17, color=GREEN, gap=24, limit=2)}
    {lines(319, 628, source["chunk_id"], size=16, color=MUTED)}
    {lines(319, 681, "Цитата проверена по фрагменту PDF", size=17, color="#b8d9ca")}
    {box(295, 763, 776, 108, "#182536", stroke="#385365")}
    {lines(319, 801, "Ваш вопрос или уточнение", size=16, color=MUTED)}
    {lines(319, 838, "Каждый вопрос ищет контекст в книге", size=17, color="#849fac")}

    {lines(1140, 54, "ПАМЯТЬ ЗАДАЧИ", size=16, color=GREEN)}
    {lines(1140, 98, "Цель", size=20)}
    {box(1137, 116, 275, 182, "#182838", stroke="#365063")}
    {lines(1150, 147, state["goal"]["value"], width=27, size=17, gap=24, limit=5)}
    {lines(1150, 277, "↗ сообщение 1", size=15, color=GREEN)}
    {lines(1140, 348, "Ограничение", size=20)}
    {box(1137, 366, 275, 124, "#182838", stroke="#365063")}
    {lines(1150, 400, constraint["value"], width=26, size=17, limit=2)}
    {lines(1150, 468, "↗ исправлено · сообщение 7", width=29, size=14, color=GREEN)}
    {lines(1140, 541, "Термин", size=20)}
    {box(1137, 559, 275, 143, "#182838", stroke="#365063")}
    {lines(1150, 590, term["key"] + ": " + term["value"], width=27, size=17, gap=23, limit=3)}
    {lines(1150, 680, "↗ сообщение 3", size=15, color=GREEN)}
    {lines(1140, 821, "История: 12 сообщений", size=17, color=MUTED)}
    </svg>'''


def make_video(report_path: Path, output: Path) -> None:
    for command in ("rsvg-convert", "ffmpeg"):
        if not shutil.which(command):
            raise RuntimeError(f"Для видео нужен {command}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("mode") != "offline_scripted" or len(report["scenarios"]) != 2:
        raise ValueError("Ожидался офлайн-отчёт двух сценариев")
    if not all(all(row["checks"].values()) for scenario in report["scenarios"]
               for row in scenario["turns"]):
        raise ValueError("В отчёте есть непройденные проверки")
    frames = [
        slide(1, "RAG-чат с памятью задачи", "Отдельный экран, история и источники",
              "2 диалога по 12 сообщений · 48/48 контрактных проверок"),
        chat_slide(report["scenarios"][0]),
        chat_slide(report["scenarios"][1]),
        slide(4, "Цель остаётся в памяти", "После выхода старых сообщений из окна модели",
              "Цель и исправление: 2/2 · Вне корпуса: отказ без ссылок"),
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="day25-video-") as temporary_directory:
        directory = Path(temporary_directory)
        for index, svg in enumerate(frames):
            source = directory / f"frame-{index}.svg"
            source.write_text(svg, encoding="utf-8")
            subprocess.run(["rsvg-convert", "--output", str(directory / f"frame-{index}.png"),
                            str(source)], check=True)
        playlist = directory / "frames.txt"
        playlist.write_text("".join(
            f"file 'frame-{index}.png'\nduration {duration}\n"
            for index, duration in enumerate((5, 8, 8, 5))
        ) + "file 'frame-3.png'\n", encoding="utf-8")
        subprocess.run([
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", str(playlist),
            "-vf", "fps=30,scale=1440:900,format=yuv420p", "-c:v", "libx264",
            "-crf", "21", "-movflags", "+faststart", "-t", "26", str(output),
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
