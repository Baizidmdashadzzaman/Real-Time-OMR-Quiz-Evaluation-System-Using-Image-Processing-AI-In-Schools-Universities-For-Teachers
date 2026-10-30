#!/usr/bin/env python3
"""Build a teacher answer key for an OMR template.

The key is deliberately separate from the sheet template: the same printed
sheet can be reused for any subject or test, and only the key changes.

Interactive use (recommended for a teacher)::

    python scripts/make_key.py --out keys/class8_bangla.json

The wizard asks for the correct option of every question and then for the
marks.  Enter keeps an existing answer when you are editing a key; ``p`` goes
back one question and ``q`` cancels without writing anything.

Edit an existing key::

    python scripts/make_key.py --from keys/class8_bangla.json \
        --out keys/class8_bangla.json

Bulk entry without prompts (positions 1-4 or the printed labels)::

    python scripts/make_key.py --out keys/class8_math.json \
        --answers "2,1,3,4,1,2,3,4,..." --marks 1

A single ``--marks`` value applies to every question; a comma-separated list
sets marks per question.  Answers and marks are validated against the template
before the file is written, and the written file is re-read and validated
again.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from omr.grade import load_key  # noqa: E402
from omr.template import Template  # noqa: E402

DEFAULT_TEMPLATE = ROOT / "templates" / "q-paper.json"
DEFAULT_OUT = ROOT / "keys" / "new_key.json"

HELP = """\
Enter        keep the shown answer (when editing)
1-5 / a-e    choose the option (also ক/খ/গ/ঘ/ঙ)
p            previous question
q            cancel without saving
?            show this help
2,1,3,4      paste several answers at once, starting at this question
"""


class Cancelled(Exception):
    """The teacher aborted the wizard."""


# --------------------------------------------------------------------- input


def _reconfigure_stdout() -> None:
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:  # pragma: no cover - old/odd console
        pass


def ask_text(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        raw = input(f"{prompt}{suffix}: ").strip()
    except EOFError as exc:
        raise Cancelled from exc
    return raw or default


def ask_yes(prompt: str, default: bool = True) -> bool:
    suffix = " [Y/n]" if default else " [y/N]"
    while True:
        try:
            raw = input(f"{prompt}{suffix}: ").strip().lower()
        except EOFError as exc:
            raise Cancelled from exc
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("  please answer y or n")


def ask_mark(prompt: str, default: float) -> float:
    while True:
        raw = ask_text(prompt, fmt_number(default))
        value = parse_mark(raw)
        if value is not None:
            return value
        print("  marks must be a number greater than or equal to 0")


# ------------------------------------------------------------------- parsing


def parse_mark(text: str) -> float | None:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return value


def fmt_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def parse_choice(text: str, labels: list[str]) -> int | None:
    """Return a zero-based option index from 1-5, a label, or a letter."""
    token = text.strip().lower()
    if not token:
        return None
    if token.isdigit():
        number = int(token)
        return number - 1 if 1 <= number <= len(labels) else None
    for index, label in enumerate(labels):
        if token == label.lower():
            return index
    if len(token) == 1 and token in "abcde"[: len(labels)]:
        return ord(token) - ord("a")
    return None


def parse_choice_list(text: str, labels: list[str]) -> list[int] | None:
    tokens = [t for t in re.split(r"[,;\s]+", text.strip()) if t]
    if len(tokens) < 2:
        return None
    out: list[int] = []
    for token in tokens:
        index = parse_choice(token, labels)
        if index is None:
            return None
        out.append(index)
    return out


def parse_mark_overrides(text: str, questions: set[int]) -> dict[int, float] | None:
    out: dict[int, float] = {}
    for token in (t for t in re.split(r"[,;]+", text) if t.strip()):
        match = re.fullmatch(r"\s*(\d+)\s*[=:]\s*(\d+(?:\.\d+)?)\s*", token)
        if match is None:
            return None
        q = int(match.group(1))
        value = parse_mark(match.group(2))
        if q not in questions or value is None:
            return None
        out[q] = value
    return out or None


# ------------------------------------------------------------------ template


def labels_for(template: Template, q: int) -> list[str]:
    return [b.label for b in template.bubbles_by_question()[q]]


def question_numbers(template: Template) -> list[int]:
    return list(template.bubbles_by_question())


def load_existing_key(path: Path, template: Template) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: could not read {path}: {exc}") from exc
    key_template = data.get("template_id")
    if key_template and key_template != template.template_id:
        raise SystemExit(
            f"error: {path} is for template {key_template!r}, "
            f"not {template.template_id!r}"
        )
    return data


# -------------------------------------------------------------------- wizard


def collect_answers(
    template: Template,
    existing: dict[int, int],
    needed: set[int],
) -> dict[int, int]:
    qnums = question_numbers(template)
    answers = {q: existing[q] for q in qnums if q in existing}
    print("\nAnswers")
    print("  Enter = keep the current answer.  p = previous, q = cancel, ? = help.")
    print("  You can paste the whole answer list at the first question, e.g. 2,1,3,4,...")

    i = 0
    while i < len(qnums):
        q = qnums[i]
        labels = labels_for(template, q)
        current = answers.get(q)
        options = "  ".join(f"{n}){lab}" for n, lab in enumerate(labels, 1))
        default = labels[current] if current is not None and current < len(labels) else ""
        raw = ask_text(f"Q{q:02d}  {options}  answer", default)

        if not raw:
            print(f"  choose a printed label or 1-{len(labels)}; q cancels")
            continue
        if raw.lower() == "q":
            raise Cancelled
        if raw.lower() == "p":
            if i == 0:
                print("  already at the first question")
            else:
                i -= 1
            continue
        if raw == "?":
            print(HELP)
            continue

        bulk = parse_choice_list(raw, labels)
        if bulk is not None:
            if i + len(bulk) > len(qnums):
                print(f"  that is {len(bulk)} answers but only {len(qnums) - i} questions remain")
                continue
            for offset, index in enumerate(bulk):
                answers[qnums[i + offset]] = index
            i += len(bulk)
            continue

        index = parse_choice(raw, labels)
        if index is None:
            print(f"  {raw!r} is not a valid option; use 1-{len(labels)} or a printed label")
            continue
        answers[q] = index
        i += 1

    missing = needed - answers.keys()
    if missing:
        raise SystemExit(f"error: no answer for questions {sorted(missing)}")
    return answers


def collect_marks(template: Template, existing: dict[int, float], interactive: bool) -> dict[int, float]:
    qnums = question_numbers(template)
    values = {round(existing[q], 6) for q in qnums if q in existing}

    if len(values) > 1:
        marks = {q: existing[q] for q in qnums if q in existing}
        print(f"\nMarks: the existing key has {len(values)} different mark values "
              f"(maximum {fmt_number(max(values))}).")
        if ask_yes("Keep the existing per-question marks?", True):
            pass
        else:
            bulk = ask_mark("Marks for every question", 1.0)
            marks = {q: bulk for q in qnums}
    else:
        default = next(iter(values)) if values else 1.0
        bulk = ask_mark("Marks for every question", default)
        marks = {q: bulk for q in qnums}

    if interactive:
        while True:
            raw = ask_text(
                "Different marks for specific questions (e.g. 3=2,7=5; Enter = none)"
            )
            if not raw:
                break
            overrides = parse_mark_overrides(raw, set(qnums))
            if overrides is None:
                print("  use question=marks pairs, e.g. 3=2,7=5")
                continue
            marks.update(overrides)
            break
    for q in qnums:
        marks.setdefault(q, 1.0)
    return marks


def summary_lines(key: dict, template: Template, path: Path) -> list[str]:
    qnums = question_numbers(template)
    total = sum(float(key["questions"][str(q)]["marks"]) for q in qnums)
    answers = "  ".join(
        f"{q}-{key['questions'][str(q)]['answer'] + 1}" for q in qnums
    )
    lines = [
        "",
        f"Key file : {path}",
        f"Key ID   : {key['key_id']}",
        f"Template : {template.template_id} ({len(qnums)} questions)",
        f"Subject  : {key['subject'] or '-'}    Set: {key['set'] or '-'}",
        f"Maximum  : {fmt_number(total)} marks",
        f"Answers  : {answers}   (option positions 1-{template.options})",
    ]
    varied = {
        q: float(key["questions"][str(q)]["marks"])
        for q in qnums
        if float(key["questions"][str(q)]["marks"]) != 1.0
    }
    if varied:
        lines.append(
            "Marks    : " + "  ".join(f"Q{q}={fmt_number(v)}" for q, v in varied.items())
        )
    return lines


# ----------------------------------------------------------------- validation


def validate_key(key: dict, template: Template, path: Path | None = None) -> list[str]:
    where = f"{path}: " if path else ""
    errors: list[str] = []
    if key.get("template_id") != template.template_id:
        errors.append(
            f"{where}template_id must be {template.template_id!r}, "
            f"got {key.get('template_id')!r}"
        )

    by_q = template.bubbles_by_question()
    questions = key.get("questions")
    if not isinstance(questions, dict):
        return errors + [f"{where}'questions' must be an object"]

    seen: set[int] = set()
    for raw_q, entry in questions.items():
        try:
            q = int(raw_q)
        except (TypeError, ValueError):
            errors.append(f"{where}question key {raw_q!r} is not a number")
            continue
        if q not in by_q:
            errors.append(f"{where}question {q} does not exist in the template")
            continue
        if q in seen:
            errors.append(f"{where}question {q} is listed twice")
        seen.add(q)
        if not isinstance(entry, dict):
            errors.append(f"{where}question {q} must be an object")
            continue

        labels = labels_for(template, q)
        answer = entry.get("answer")
        if isinstance(answer, bool) or not isinstance(answer, int):
            errors.append(f"{where}question {q} answer must be a whole number 0-{len(labels) - 1}")
        elif not 0 <= answer < len(labels):
            errors.append(
                f"{where}question {q} answer {answer} is outside 0-{len(labels) - 1} "
                f"({', '.join(labels)})"
            )

        marks = entry.get("marks", 1)
        if isinstance(marks, bool) or not isinstance(marks, (int, float)):
            errors.append(f"{where}question {q} marks must be a number")
        elif not math.isfinite(float(marks)) or float(marks) < 0:
            errors.append(f"{where}question {q} marks must be >= 0")

    missing = sorted(set(by_q) - seen)
    if missing:
        errors.append(f"{where}missing answers for questions {missing}")

    policy = key.get("policy", {})
    if policy.get("multi") not in {"wrong", "partial", "review"}:
        errors.append(f"{where}policy.multi must be wrong, partial, or review")
    negative = policy.get("negative_per_wrong", 0)
    if isinstance(negative, bool) or not isinstance(negative, (int, float)):
        errors.append(f"{where}policy.negative_per_wrong must be a number")
    elif not math.isfinite(float(negative)) or float(negative) < 0:
        errors.append(f"{where}policy.negative_per_wrong must be >= 0")
    return errors


# ---------------------------------------------------------------------- write


def write_key(path: Path, key: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def build_key(
    template: Template,
    answers: dict[int, int],
    marks: dict[int, float],
    key_id: str,
    subject: str,
    set_name: str,
    policy: dict,
) -> dict:
    return {
        "key_id": key_id,
        "template_id": template.template_id,
        "subject": subject,
        "set": set_name,
        "questions": {
            str(q): {
                "answer": int(answers[q]),
                "marks": (
                    int(marks[q])
                    if float(marks[q]).is_integer()
                    else float(marks[q])
                ),
            }
            for q in question_numbers(template)
        },
        "policy": policy,
    }


# ------------------------------------------------------------------------ CLI


def main() -> int:
    _reconfigure_stdout()
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    ap.add_argument("--out", default=None, help=f"key path to write (default: {DEFAULT_OUT})")
    ap.add_argument("--from", dest="from_key", default=None,
                    help="existing key to edit; its answers, marks and metadata are preloaded")
    ap.add_argument("--key-id", default=None, help="key_id stored in the file")
    ap.add_argument("--subject", default=None)
    ap.add_argument("--set", dest="set_name", default=None)
    ap.add_argument("--answers", default=None,
                    help="comma-separated correct options (1-4 or labels); skips prompts")
    ap.add_argument("--marks", default=None,
                    help="one mark value for all questions, or a comma-separated list")
    ap.add_argument("--negative", type=float, default=None,
                    help="marks deducted per wrong answer (default: existing or 0)")
    ap.add_argument("--multi", choices=["wrong", "partial", "review"], default=None,
                    help="policy for more than one mark (default: existing or wrong)")
    ap.add_argument("--force", action="store_true", help="overwrite the output without asking")
    ap.add_argument("--dry-run", action="store_true", help="validate and print, do not write")
    args = ap.parse_args()

    template_path = Path(args.template)
    if not template_path.exists():
        print(f"error: template not found: {template_path}", file=sys.stderr)
        return 2
    template = Template.load(template_path)
    qnums = question_numbers(template)
    needed = set(qnums)

    existing_key: dict = {}
    if args.from_key:
        from_path = Path(args.from_key)
        if not from_path.exists():
            print(f"error: key not found: {from_path}", file=sys.stderr)
            return 2
        existing_key = load_existing_key(from_path, template)
    existing_q = existing_key.get("questions", {})
    existing_answers: dict[int, int] = {}
    existing_marks: dict[int, float] = {}
    for raw_q, entry in existing_q.items():
        if not isinstance(entry, dict) or not str(raw_q).isdigit():
            continue
        q = int(raw_q)
        try:
            existing_answers[q] = int(entry["answer"])
        except (KeyError, TypeError, ValueError):
            pass
        try:
            existing_marks[q] = float(entry.get("marks", 1))
        except (TypeError, ValueError):
            pass
    existing_policy = existing_key.get("policy", {})

    if args.answers is None:
        interactive = True
        if not sys.stdin.isatty():
            print(
                "error: interactive entry needs a terminal.\n"
                "Use --answers \"2,1,3,4,...\" for non-interactive entry.",
                file=sys.stderr,
            )
            return 2
        out = Path(ask_text("Output key file", str(Path(args.out) if args.out else DEFAULT_OUT)))
    else:
        interactive = False
        out = Path(args.out) if args.out else DEFAULT_OUT
        if out.exists() and not args.force:
            print(f"error: {out} exists; pass --force to replace it", file=sys.stderr)
            return 2

    if out.exists() and not args.force and interactive:
        print(f"note: {out} exists and will be replaced when you save.")

    key_id = args.key_id or existing_key.get("key_id") or out.stem
    subject = args.subject if args.subject is not None else existing_key.get("subject", "")
    set_name = args.set_name if args.set_name is not None else existing_key.get("set", "A")
    if interactive:
        key_id = ask_text("Key ID", key_id)
        subject = ask_text("Subject", subject)
        set_name = ask_text("Set", set_name or "A")

    try:
        if args.answers is None:
            answers = collect_answers(template, existing_answers, needed)
        else:
            tokens = [t for t in re.split(r"[,;\s]+", args.answers.strip()) if t]
            if len(tokens) != len(qnums):
                print(
                    f"error: --answers has {len(tokens)} values but the template has "
                    f"{len(qnums)} questions",
                    file=sys.stderr,
                )
                return 2
            answers = {}
            for q, token in zip(qnums, tokens):
                index = parse_choice(token, labels_for(template, q))
                if index is None:
                    print(
                        f"error: answer {token!r} for question {q} is not 1-"
                        f"{len(labels_for(template, q))} or a printed label",
                        file=sys.stderr,
                    )
                    return 2
                answers[q] = index

        if args.marks is None:
            if interactive:
                marks = collect_marks(template, existing_marks, True)
            else:
                marks = {q: float(existing_marks.get(q, 1)) for q in qnums}
        else:
            mark_tokens = [t for t in re.split(r"[,;\s]+", args.marks.strip()) if t]
            values = [parse_mark(t) for t in mark_tokens]
            if not values or any(v is None for v in values):
                print("error: --marks must be a number or a list of numbers >= 0", file=sys.stderr)
                return 2
            clean = [float(v) for v in values if v is not None]
            if len(clean) == 1:
                marks = {q: clean[0] for q in qnums}
            elif len(clean) == len(qnums):
                marks = dict(zip(qnums, clean))
            else:
                print(
                    f"error: --marks has {len(clean)} values; use 1 value or "
                    f"{len(qnums)} values",
                    file=sys.stderr,
                )
                return 2
    except Cancelled:
        print("\ncancelled; nothing was written")
        return 130

    try:
        existing_negative = float(existing_policy.get("negative_per_wrong", 0.0))
    except (TypeError, ValueError):
        existing_negative = 0.0
    policy = {
        "multi": args.multi or existing_policy.get("multi", "wrong"),
        "blank": 0,
        "negative_per_wrong": args.negative if args.negative is not None else existing_negative,
        "partial_multi": 0.0,
    }
    if policy["negative_per_wrong"] < 0:
        print("error: --negative must be >= 0", file=sys.stderr)
        return 2

    key = build_key(template, answers, marks, key_id, subject, set_name, policy)
    errors = validate_key(key, template)
    if errors:
        print("error: the key is not valid:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 2

    for line in summary_lines(key, template, out):
        print(line)

    if args.dry_run:
        print("\n" + json.dumps(key, ensure_ascii=False, indent=2))
        print("\ndry run: not written")
        return 0

    if interactive and not ask_yes(f"\nSave this key to {out}?", True):
        print("not saved")
        return 0

    write_key(out, key)

    # Prove the on-disk file is readable and valid before reporting success.
    try:
        reloaded = json.loads(out.read_text(encoding="utf-8"))
        load_key(out, template)
    except Exception as exc:  # pragma: no cover - disk/permission problems
        print(f"error: wrote {out} but could not read it back: {exc}", file=sys.stderr)
        return 1
    errors = validate_key(reloaded, template, out)
    if errors:
        print("error: the written key failed validation:", file=sys.stderr)
        for error in errors:
            print(f"  {error}", file=sys.stderr)
        return 1

    print(f"\nsaved: {out}")
    print("use it with:")
    print(f'  python scripts/live_camera_with_audio.py --key "{out}"')
    print(f'  python scripts/read_image.py "sheet.jpg" --key "{out}"')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
