"""Measures answer quality on a fictional handbook, so tuning is based on numbers.

python eval/evaluate.py             compare the search settings (takes about a minute)
python eval/evaluate.py --answers   also score the answers from the current model (slower)
"""
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from qa_system import NOT_FOUND, DocumentQA

EVAL_DIR = Path(__file__).parent
QUESTIONS = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
ANSWERABLE = [q for q in QUESTIONS if not q.get("unanswerable")]
UNANSWERABLE = [q for q in QUESTIONS if q.get("unanswerable")]


def contains_any(text, expected):
    return any(phrase.lower() in text.lower() for phrase in expected)


def score_search(qa):
    """how often do the chunks given to the model contain the right fact?"""
    hits = 0
    misses = []
    start = time.time()
    for item in ANSWERABLE:
        text = " ".join(doc.page_content for doc in qa.retrieve(item["question"]))
        if contains_any(text, item["expected"]):
            hits += 1
        else:
            misses.append(item["question"])
    seconds = (time.time() - start) / len(ANSWERABLE)
    return hits, misses, seconds


def score_answers(qa):
    """how often is the final answer right, and does it decline when it should?"""
    correct = 0
    declined = 0
    wrong = []
    start = time.time()

    for item in QUESTIONS:
        stream, _ = qa.ask(item["question"])
        answer = "".join(stream)
        if item.get("unanswerable"):
            passed = NOT_FOUND.lower().rstrip(".") in answer.lower()
            declined += passed
        else:
            passed = contains_any(answer, item["expected"])
            correct += passed
        if not passed:
            wrong.append((item["question"], answer))

    seconds = (time.time() - start) / len(QUESTIONS)
    return correct, declined, wrong, seconds


def main():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as folder:
        qa = DocumentQA(persist_dir=folder)
        qa.add_document(EVAL_DIR / "handbook.txt")

        print(f"\nSearch: right fact found in the top chunks ({len(ANSWERABLE)} questions)")
        settings = [
            ("meaning search only", False, False),
            ("+ keyword search", True, False),
            ("+ keyword search + reranker", True, True),
        ]
        for name, use_hybrid, use_rerank in settings:
            qa.use_hybrid = use_hybrid
            qa.use_rerank = use_rerank
            hits, misses, seconds = score_search(qa)
            print(f"  {name:30} {hits}/{len(ANSWERABLE)}   {seconds:.2f}s per question")
            for question in misses:
                print(f"      missed: {question}")

        if "--answers" in sys.argv:
            correct, declined, wrong, seconds = score_answers(qa)
            print(f"\nAnswers from the current model ({seconds:.1f}s per question)")
            print(f"  Correct answer:    {correct}/{len(ANSWERABLE)}")
            print(f"  Declined properly: {declined}/{len(UNANSWERABLE)}")
            for question, answer in wrong:
                print(f"\n  FAILED: {question}\n  Said: {answer[:300]}")


if __name__ == "__main__":
    main()
