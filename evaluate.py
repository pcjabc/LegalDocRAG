"""对固定问题集执行可复现的检索与拒答评测。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from legaldoc_rag.config import Settings
from legaldoc_rag.service import LegalDocService

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="评测 LegalDoc RAG 的检索与拒答")
    parser.add_argument("--questions", type=Path, default=ROOT / "data" / "evaluation" / "questions.jsonl")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    settings = Settings.from_env(ROOT)
    service = LegalDocService(settings)
    service.build_index(sorted((ROOT / "data" / "samples").glob("*.md")))
    cases = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines() if line.strip()]

    hit_count = 0
    reciprocal_rank_total = 0.0
    refusal_correct = 0
    refusal_total = 0
    durations: list[float] = []

    print("id\t命中\t排名\t拒答正确\t耗时(ms)\t问题")
    for case in cases:
        started = time.perf_counter()
        answer = service.ask(case["question"], top_k=args.top_k)
        durations.append((time.perf_counter() - started) * 1000)
        rank = next(
            (
                position
                for position, hit in enumerate(answer.sources, 1)
                if hit.chunk.source_name == case["expected_source"] and hit.chunk.section == case["expected_section"]
            ),
            None,
        )
        hit = (rank is not None and not answer.abstained) if case["answerable"] else answer.abstained
        if case["answerable"]:
            hit_count += int(hit)
            reciprocal_rank_total += 1 / rank if rank and not answer.abstained else 0
        else:
            refusal_total += 1
            refusal_correct += int(answer.abstained)
        print(
            f"{case['id']}\t{hit}\t{rank or '-'}\t"
            f"{answer.abstained == (not case['answerable'])}\t{durations[-1]:.2f}\t{case['question']}"
        )

    answerable_total = sum(1 for case in cases if case["answerable"])
    print("\n汇总")
    print(f"Hit@{args.top_k}: {hit_count / answerable_total:.3f} ({hit_count}/{answerable_total})")
    print(f"MRR: {reciprocal_rank_total / answerable_total:.3f}")
    print(f"拒答准确率: {refusal_correct / refusal_total:.3f} ({refusal_correct}/{refusal_total})")
    print(f"平均查询耗时: {sum(durations) / len(durations):.2f} ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
