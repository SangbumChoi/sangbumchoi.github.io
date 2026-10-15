"""A tiny RL simulation: compare four post-training methods on a small transformer that does 3-digit addition and subtraction.

- SFT distillation: copy the teacher's model answer verbatim (off-policy; the student never solves anything itself).
- RLVR (GRPO): the student writes 8 answers per problem, and the verifier returns a single pass/fail per answer.
- OPD: for every character the student wrote, the teacher gives "what I would write here" probabilities, and the student matches them.
- MOPD: OPD where each subject (addition / subtraction) is graded by that subject's specialist teacher.

The model writes the answer ones digit first (the order of long addition by hand = a tiny reasoning trace).
The verifier only checks that all five characters (four digits + end marker) match the answer.

    python3 rlsim.py all                       # train teachers, then the four experiments; results in results/*.json
    python3 rlsim.py single --seeds 1 --steps 100   # quick check
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

torch.set_num_threads(4)

VOCAB = list("0123456789+-=$")  # "$" = end of answer
STOI = {c: i for i, c in enumerate(VOCAB)}
PROMPT_LEN, ANSWER_LEN = 8, 5  # after "123+045=": four digits ones-first, then "$"
SEQ_LEN = PROMPT_LEN + ANSWER_LEN
BATCH = 256  # answers the student writes per step; identical for every method
GROUP = 8  # RLVR: answers per problem (32 problems x 8)
LR = 5e-4
LOG_EVERY = 20
WARM_TARGET = 0.30  # student starting point: 30% accuracy (mimics pre-training + a little SFT)
OUT = Path(__file__).resolve().parent / "results"


# ---------------------------------------------------------------- problems and verifier


def problems(n: int, op: str, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """n problems (prompts) and answers as token ids. Subtraction always subtracts the smaller number from the larger."""
    a, b = rng.integers(0, 1000, n), rng.integers(0, 1000, n)
    if op == "-":
        a, b = np.maximum(a, b), np.minimum(a, b)
    c = a + b if op == "+" else a - b
    prompts = [[STOI[ch] for ch in f"{x:03d}{op}{y:03d}="] for x, y in zip(a, b)]
    answers = [[STOI[ch] for ch in f"{z:04d}"[::-1] + "$"] for z in c]
    return torch.tensor(prompts), torch.tensor(answers)


def mixed(n: int, ops: list[str], rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """n problems split evenly across subjects. The third value is each row's subject."""
    per = n // len(ops)
    parts = [problems(per, op, rng) for op in ops]
    subjects = [op for op in ops for _ in range(per)]
    return torch.cat([p for p, _ in parts]), torch.cat([a for _, a in parts]), subjects


def correct(seq: torch.Tensor, answers: torch.Tensor) -> torch.Tensor:
    """Verifier: 1 if the written answer matches the reference character for character, else 0."""
    return (seq[:, PROMPT_LEN:] == answers).all(1).float()


# ---------------------------------------------------------------- model


class TinyGPT(nn.Module):
    def __init__(self, layers: int, width: int, heads: int = 4):
        super().__init__()
        self.tok = nn.Embedding(len(VOCAB), width)
        self.pos = nn.Embedding(SEQ_LEN, width)
        block = nn.TransformerEncoderLayer(
            width, heads, 4 * width, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, len(VOCAB))
        self.register_buffer("mask", nn.Transformer.generate_square_subsequent_mask(SEQ_LEN), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        h = self.tok(x) + self.pos.weight[:t]
        h = self.blocks(h, mask=self.mask[:t, :t], is_causal=True)
        return self.head(self.norm(h))


def student() -> TinyGPT:
    return TinyGPT(layers=2, width=64)


def teacher() -> TinyGPT:
    return TinyGPT(layers=4, width=128)


def answer_logits(model: TinyGPT, seq: torch.Tensor) -> torch.Tensor:
    """The prediction made just before each of the five answer characters (batch, 5, vocab)."""
    return model(seq)[:, PROMPT_LEN - 1 : -1]


@torch.no_grad()
def write(model: TinyGPT, prompts: torch.Tensor, greedy: bool = False) -> torch.Tensor:
    """The model writes the answer one character at a time. greedy=False samples from its probabilities (temperature 1)."""
    seq = prompts
    for _ in range(ANSWER_LEN):
        logits = model(seq)[:, -1]
        nxt = logits.argmax(-1) if greedy else torch.multinomial(logits.softmax(-1), 1).squeeze(1)
        seq = torch.cat([seq, nxt[:, None]], 1)
    return seq


def accuracy(model: TinyGPT, test: tuple[torch.Tensor, torch.Tensor]) -> float:
    prompts, answers = test
    return correct(write(model, prompts, greedy=True), answers).mean().item()


# ---------------------------------------------------------------- one step of each method


def update(opt: torch.optim.Optimizer, model: TinyGPT, loss: torch.Tensor) -> None:
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()


def step_sft(model: TinyGPT, opt, prompts: torch.Tensor, targets: torch.Tensor) -> dict:
    """Copy the given answer character by character (cross-entropy)."""
    seq = torch.cat([prompts, targets], 1)
    loss = F.cross_entropy(answer_logits(model, seq).reshape(-1, len(VOCAB)), targets.reshape(-1))
    update(opt, model, loss)
    return {}


def teacher_tokens(prompts: torch.Tensor, subjects: list[str], teachers: dict[str, TinyGPT]) -> torch.Tensor:
    """For each subject, the model answer written by that subject's teacher (greedy)."""
    targets = torch.empty(len(prompts), ANSWER_LEN, dtype=torch.long)
    for op, model in teachers.items():
        rows = torch.tensor([s == op for s in subjects])
        if rows.any():
            targets[rows] = write(model, prompts[rows], greedy=True)[:, PROMPT_LEN:]
    return targets


def step_rlvr(model: TinyGPT, opt, prompts: torch.Tensor, answers: torch.Tensor) -> dict:
    """GRPO: write 8 answers per problem; raise the probability of the characters in correct answers, lower it in wrong ones.
    If all 8 are right or all 8 are wrong there is nothing to compare, so that problem teaches nothing."""
    prompts, answers = prompts.repeat_interleave(GROUP, 0), answers.repeat_interleave(GROUP, 0)
    seq = write(model, prompts)
    reward = correct(seq, answers).view(-1, GROUP)
    advantage = (reward - reward.mean(1, keepdim=True)) / (reward.std(1, keepdim=True) + 1e-6)
    logp = answer_logits(model, seq).log_softmax(-1).gather(-1, seq[:, PROMPT_LEN:, None]).squeeze(-1)
    update(opt, model, -(advantage.view(-1, 1) * logp).mean())
    return {"signal": (reward.std(1) > 0).float().mean().item(), "reward": reward.mean().item()}


def step_opd(model: TinyGPT, opt, prompts: torch.Tensor, subjects: list[str], teachers: dict[str, TinyGPT]) -> dict:
    """OPD: the student writes; keeping the student's own prefix, shrink the gap between the teacher's and the student's
    distribution at every character (reverse KL over the whole vocabulary). A different teacher per subject makes it MOPD."""
    seq = write(model, prompts)
    s = answer_logits(model, seq).log_softmax(-1)
    t = torch.empty_like(s)
    with torch.no_grad():
        for op, model_t in teachers.items():
            rows = torch.tensor([x == op for x in subjects])
            if rows.any():
                t[rows] = answer_logits(model_t, seq[rows]).log_softmax(-1)
    kl = (s.exp() * (s - t)).sum(-1)
    update(opt, model, kl.mean())
    return {"signal": 1.0, "kl": kl.mean().item()}


# ---------------------------------------------------------------- training loops


def sft_until(model: TinyGPT, ops: list[str], target: float, rng, val: dict, check_every: int, max_steps: int) -> dict:
    """SFT on the true answers until accuracy reaches target (used to build teachers and the student's starting point)."""
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(max_steps + 1):
        if step % check_every == 0:
            accs = {op: accuracy(model, val[op]) for op in ops}
            if np.mean(list(accs.values())) >= target or step == max_steps:
                return {"steps": step, **accs}
        prompts, answers, _ = mixed(BATCH, ops, rng)
        step_sft(model, opt, prompts, answers)
    raise AssertionError


def post_train(method: str, model: TinyGPT, plan: list[tuple[int, list[str]]], teachers: dict, tests: dict, seed: int) -> dict:
    """plan = [(steps, subjects), ...]; two stages make a cascade (finish one subject, then the next)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    curve, signals, step, started = [], [], 0, time.time()
    schedule = [ops for steps, ops in plan for _ in range(steps)]
    for step in range(len(schedule) + 1):
        if step % LOG_EVERY == 0:
            point = {"step": step, **{op: accuracy(model, tests[op]) for op in tests}}
            point["signal"] = float(np.mean(signals)) if signals else None
            curve.append(point)
            signals = []
        if step == len(schedule):
            break
        ops = schedule[step]
        if method == "rlvr":
            prompts, answers, _ = mixed(BATCH // GROUP, ops, rng)
            stats = step_rlvr(model, opt, prompts, answers)
        else:
            prompts, answers, subjects = mixed(BATCH, ops, rng)
            if method == "sft":
                stats = step_sft(model, opt, prompts, teacher_tokens(prompts, subjects, teachers))
            else:
                stats = step_opd(model, opt, prompts, subjects, teachers)
        if "signal" in stats:
            signals.append(stats["signal"])
    return {"curve": curve, "seconds": round(time.time() - started, 1)}


# ---------------------------------------------------------------- experiments


def test_sets() -> dict:
    rng = np.random.default_rng(12345)
    return {op: problems(2000, op, rng) for op in "+-"}


def val_sets() -> dict:
    rng = np.random.default_rng(999)
    return {op: problems(1000, op, rng) for op in "+-"}


def load_or_train_teachers() -> dict[str, TinyGPT]:
    """An addition specialist, a subtraction specialist (each learns only its own subject), and an under-trained addition teacher (~70%)."""
    folder = OUT / "teachers"
    folder.mkdir(parents=True, exist_ok=True)
    specs = {"add": (["+"], 0.995, 1), "sub": (["-"], 0.995, 2), "add-weak": (["+"], 0.70, 3)}
    teachers, report = {}, {}
    for name, (ops, target, seed) in specs.items():
        torch.manual_seed(seed)
        model = teacher()
        path = folder / f"{name}.pt"
        if path.exists():
            model.load_state_dict(torch.load(path))
        else:
            started = time.time()
            info = sft_until(model, ops, target, np.random.default_rng(seed), val_sets(), 25 if target < 0.9 else 100, 30000)
            torch.save(model.state_dict(), path)
            print(f"teacher {name}: {info} in {time.time() - started:.0f}s", flush=True)
        teachers[name] = model
        report[name] = {op: accuracy(model, t) for op, t in test_sets().items()}
    (folder / "accuracy.json").write_text(json.dumps(report, indent=1))
    print("teacher test accuracy:", report, flush=True)
    return teachers


def warm_student(ops: list[str], seed: int) -> tuple[TinyGPT, dict]:
    torch.manual_seed(seed)
    model = student()
    info = sft_until(model, ops, WARM_TARGET, np.random.default_rng(1000 + seed), val_sets(), 5, 5000)
    return model, info


def example_card(model: TinyGPT, add_teacher: TinyGPT) -> dict:
    """One problem the student gets wrong: the verifier gives one pass/fail, the teacher gives probabilities per digit."""
    prompts, answers = test_sets()["+"]
    seq = write(model, prompts[:400], greedy=True)
    wrong = (correct(seq, answers[:400]) == 0).nonzero()[0, 0].item()
    one = seq[wrong : wrong + 1]
    with torch.no_grad():
        t = answer_logits(add_teacher, one).softmax(-1)[0]
        s = answer_logits(model, one).softmax(-1)[0]
    places = ["ones", "tens", "hundreds", "thousands"]
    rows = []
    for i, place in enumerate(places):
        tok = one[0, PROMPT_LEN + i].item()
        rows.append(
            {
                "place": place,
                "student_wrote": VOCAB[tok],
                "true": VOCAB[answers[wrong, i].item()],
                "student_prob": round(s[i, tok].item(), 3),
                "teacher_prob_of_that": round(t[i, tok].item(), 3),
                "teacher_would_write": VOCAB[t[i].argmax().item()],
            }
        )
    text = "".join(VOCAB[i] for i in prompts[wrong].tolist())
    return {
        "problem": text,
        "true_answer": "".join(VOCAB[i] for i in answers[wrong, :4].tolist())[::-1].lstrip("0") or "0",
        "student_answer": "".join(VOCAB[i] for i in one[0, PROMPT_LEN : PROMPT_LEN + 4].tolist())[::-1].lstrip("0") or "0",
        "verifier": 0,
        "places": rows,
    }


def experiment(name: str, seeds: int, steps: int, teachers: dict) -> dict:
    tests = test_sets()
    add, sub, weak = teachers["add"], teachers["sub"], teachers["add-weak"]
    if name == "multi":
        runs = {
            "mix-rlvr": ("rlvr", [(steps, ["+", "-"])], {}),
            "cascade-rlvr": ("rlvr", [(steps // 2, ["+"]), (steps // 2, ["-"])], {}),
            "mopd": ("opd", [(steps, ["+", "-"])], {"+": add, "-": sub}),
            "opd-add-teacher-only": ("opd", [(steps, ["+", "-"])], {"+": add, "-": add}),
        }
    else:
        runs = {
            "single": {
                "sft": ("sft", [(steps, ["+"])], {"+": add}),
                "rlvr": ("rlvr", [(steps, ["+"])], {}),
                "opd": ("opd", [(steps, ["+"])], {"+": add}),
            },
            "cold": {
                "rlvr": ("rlvr", [(steps, ["+"])], {}),
                "opd": ("opd", [(steps, ["+"])], {"+": add}),
            },
            "weak": {
                "rlvr": ("rlvr", [(steps, ["+"])], {}),
                "opd-weak": ("opd", [(steps, ["+"])], {"+": weak}),
                "sft-weak": ("sft", [(steps, ["+"])], {"+": weak}),
            },
        }[name]
    ops = ["+", "-"] if name == "multi" else ["+"]
    if name != "multi":
        tests = {"+": tests["+"]}
    result = {"steps": steps, "seeds": seeds, "batch": BATCH, "group": GROUP, "lr": LR, "warm": [], "runs": {}}
    for seed in range(seeds):
        if name == "cold":
            torch.manual_seed(seed)
            start, info = student(), {"steps": 0, "+": accuracy(student(), tests["+"])}
        else:
            start, info = warm_student(ops, seed)
        result["warm"].append(info)
        if name == "single" and seed == 0:
            result["example"] = example_card(start, add)
        for run, (method, plan, run_teachers) in runs.items():
            out = post_train(method, copy.deepcopy(start), plan, run_teachers, tests, seed)
            result["runs"].setdefault(run, []).append(out)
            last = out["curve"][-1]
            print(f"{name}/{run} seed {seed}: start {info} -> {{{', '.join(f'{op}: {last[op]:.3f}' for op in tests)}}} in {out['seconds']}s", flush=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(result, indent=1, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=["all", "teachers", "single", "cold", "weak", "multi"])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    teachers = load_or_train_teachers()
    names = ["single", "cold", "weak", "multi"] if args.which == "all" else [] if args.which == "teachers" else [args.which]
    for name in names:
        experiment(name, args.seeds, args.steps, teachers)


if __name__ == "__main__":
    main()
