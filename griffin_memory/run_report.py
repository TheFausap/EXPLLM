"""Plan a run budget, estimate wall-clock time, and assemble a run report.

Three subcommands:

* `plan`     — turn corpus targets, batch shape and epochs into steps/tokens.
* `estimate` — read `train.jsonl` and extrapolate throughput to a planned budget.
* `build`    — collect metrics, evaluations and generation samples into one report.

Everything here is read-only with respect to checkpoints and corpora.
"""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import statistics


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_log(path):
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def format_targets(count):
    return f"{count / 1e9:.2f}B" if count >= 1e9 else f"{count / 1e6:.2f}M" if count >= 1e6 else f"{count / 1e3:.1f}k"


def find(run_dir, *relative):
    for candidate in relative:
        path = Path(run_dir) / candidate
        if path.is_file():
            return path
    return None


def plan(args):
    manifest = load_json(Path(args.data) / "manifest.json")
    config = load_json(args.config)
    if "train" not in manifest.get("stats", {}):
        raise ValueError("Corpus has no train split")
    targets = manifest["stats"]["train"]["prediction_targets"]
    tokens_per_update = args.batch_size * args.accumulation_steps * config["chunk_size"]
    if tokens_per_update <= 0:
        raise ValueError("batch-size, accumulation-steps and chunk-size must be positive")
    full_pass_steps = math.ceil(targets / tokens_per_update)
    if args.max_steps and args.token_budget:
        raise ValueError("Pass either --max-steps or --token-budget, not both")
    planned_steps = (args.max_steps or math.ceil(args.token_budget / tokens_per_update)) if \
        (args.max_steps or args.token_budget) else math.ceil(args.epochs * full_pass_steps)
    if planned_steps <= 0:
        raise ValueError("Planned steps must be positive")
    planned_targets = planned_steps * tokens_per_update
    result = {
        "data": str(args.data), "config": str(args.config),
        "chunk_size": config["chunk_size"], "batch_size": args.batch_size,
        "accumulation_steps": args.accumulation_steps, "tokens_per_update": tokens_per_update,
        "train_documents": manifest["stats"]["train"]["documents"],
        "train_targets": targets,
        "validation_documents": manifest["stats"].get("val", {}).get("documents", 0),
        "validation_targets": manifest["stats"].get("val", {}).get("prediction_targets", 0),
        "test_documents": manifest["stats"].get("test", {}).get("documents", 0),
        "full_pass_steps": full_pass_steps, "epochs_requested": args.epochs,
        "token_budget": args.token_budget, "max_steps_requested": args.max_steps,
        "planned_steps": planned_steps,
        "planned_targets_upper_bound": planned_targets,
        "epochs_equivalent": planned_targets / targets if targets else None,
        "notes": [],
    }
    if planned_steps < full_pass_steps:
        result["notes"].append(f"{planned_steps} steps train on {planned_targets / targets:.0%} of one pass over the "
                               "training split; raise --token-budget/--max-steps or lower the corpus limit for a "
                               "full epoch.")
    if targets < 200_000_000:
        result["notes"].append(f"Only {format_targets(targets)} training targets: enough for a mechanics check or a "
                               "small model, not for fluent modern prose. Increase --train-limit or epochs.")
    if manifest.get("tokenizer_training", {}).get("reused"):
        result["notes"].append("The BPE vocabulary was reused from another corpus; token ids stay compatible, "
                               "but a strict full-state resume onto that other corpus is still rejected.")
    result["notes"].append("Actual logged tokens per update are lower than the upper bound because padded and short "
                           "final chunks do not contribute targets.")
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def estimate(args):
    records = read_log(args.log)
    if not records:
        raise ValueError("Training log is empty")
    timed = [record for record in records if record.get("tokens_per_second")]
    steps = [record for record in records if record.get("step")]
    result = {"log": str(args.log), "records": len(records),
              "steps_logged": max((record["step"] for record in steps), default=0),
              "tokens_logged": max((record.get("tokens", 0) for record in records), default=0)}
    if not timed:
        result["notes"] = ["No throughput records (log-every may be larger than the steps run)."]
        print(json.dumps(result, indent=2))
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    window = timed[-20:]
    rate = statistics.median(record["tokens_per_second"] for record in window)
    tokens_per_step = statistics.median(max(1, record.get("tokens", 1)) / record["step"] for record in window)
    seconds_per_step = tokens_per_step / rate
    result.update({
        "median_tokens_per_second": rate, "tokens_per_update": tokens_per_step,
        "seconds_per_update": seconds_per_step,
        "timed_records": len(timed), "window_records": len(window),
        "train_loss_first": next((record["train_loss"] for record in records if "train_loss" in record), None),
        "train_loss_last": next((record["train_loss"] for record in reversed(records) if "train_loss" in record), None),
        "memory_entries_last": next((record.get("memory_entries") for record in reversed(records)
                                     if record.get("memory_entries") is not None), None),
        "validation": [[record["step"], record["val_loss"], record["val_perplexity"]] for record in records
                       if "val_loss" in record],
    })
    if args.planned_steps:
        result["planned_steps"] = args.planned_steps
        result["planned_seconds"] = args.planned_steps * seconds_per_step
        result["planned_hours"] = result["planned_seconds"] / 3600
        result["planned_targets"] = args.planned_steps * tokens_per_step
    result["notes"] = ["Extrapolated from the median of the last "
                       f"{len(window)} throughput records; a longer document mix, thermal throttling or a shared GPU "
                       "will change it."]
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def strip_heading(markdown):
    lines = markdown.splitlines()
    return "\n".join(lines[1:]).lstrip("\n") if lines and lines[0].startswith("# ") else markdown


def describe_split(stats, name):
    if not stats:
        return f"- {name}: not prepared"
    return (f"- {name}: {stats.get('documents', 0)} documents, {stats.get('tokens', 0):,} tokens, "
            f"{stats.get('prediction_targets', 0):,} prediction targets")


def build(args):
    run_dir = Path(args.run_dir)

    def resolve(explicit, *relative):
        return Path(explicit) if explicit else find(run_dir, *relative)

    paths = {
        "run_json": resolve(args.run_json, "train/run.json", "run.json"),
        "config": resolve(args.config, "train/config.json", "config.json"),
        "log": resolve(args.log, "train/train.jsonl", "train.jsonl"),
        "plan": resolve(args.plan, "driver/plan.json", "plan.json"),
        "estimate": resolve(args.estimate, "driver/estimate.json"),
        "commands": resolve(args.commands, "driver/commands.log", "commands.log"),
        "corpus": Path(args.data) / "manifest.json" if args.data else
                  find(run_dir, "corpus/manifest.json", "data/manifest.json"),
        "sources": Path(args.data) / "sources.json" if args.data else find(run_dir, "corpus/sources.json"),
    }
    samples_paths = [Path(item) for item in (args.samples or [])]
    if not samples_paths:
        sample_dir = find(run_dir, "samples/samples.json", "samples.json")
        if sample_dir:
            samples_paths = [sample_dir]
    evaluations = [Path(item) for item in (args.eval_ or [])]
    if not evaluations:
        evaluations = sorted((run_dir / "eval").glob("*.json")) if (run_dir / "eval").is_dir() else []
    missing = [name for name in ("run_json", "log") if not paths[name]]
    if missing:
        raise ValueError(f"Cannot build a report without: {', '.join(missing)} (searched {run_dir})")
    options = load_json(paths["run_json"])
    outputs = [("checkpoints", run_dir / "train" / "last.pt"), ("best checkpoint", run_dir / "train" / "best.pt"),
               ("training log", paths["log"]), ("generation samples", samples_paths[0] if samples_paths else None),
               ("driver log", run_dir / "driver" / "commands.log")]
    lines = [f"# Run report — {args.title or run_dir.name}", "",
             f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from `{run_dir}`.", "",
             "Key outputs: " + ", ".join(f"`{path}` ({label})" for label, path in outputs if path), ""]
    if paths["corpus"]:
        manifest = load_json(paths["corpus"])
        lines += ["## Corpus", "",
                  f"- prepared dataset: `{paths['corpus'].parent}` (format version {manifest.get('version')}, "
                  f"dtype {manifest.get('dtype')}, vocab {manifest.get('vocab_size')})",
                  describe_split(manifest.get("stats", {}).get("train"), "train"),
                  describe_split(manifest.get("stats", {}).get("val"), "val"),
                  describe_split(manifest.get("stats", {}).get("test"), "test")]
        if paths["sources"] and paths["sources"].is_file():
            sources = load_json(paths["sources"])
            limits = sources.get("requested_limits", {})
            lines.append(f"- source: {sources.get('dataset')} ({sources.get('source')}); declared license "
                         f"{sources.get('declared_license')}; requested limits {limits}; seed {sources.get('seed')}")
            counters = sources.get("filter_statistics") or {}
            if counters:
                lines.append("- filtering: " + "; ".join(
                    f"{split}: {values.get('written', 0)} kept, {values.get('duplicates_removed', 0)} duplicates, "
                    f"{values.get('empty_removed', 0)} empty" for split, values in sorted(counters.items())))
            lines.append("- downstream users must verify the source license and the rights in the underlying works")
        lines.append("")
    lines += ["## Training", ""]
    training_options = options.get("options", {})
    lines.append(f"- device `{options.get('device')}`, torch {options.get('torch')}, seed {training_options.get('seed')}, "
                 f"precision {training_options.get('precision')}, deterministic {training_options.get('deterministic')}")
    lines.append(f"- batch {training_options.get('batch_size')} x accumulation "
                 f"{training_options.get('accumulation_steps')}, learning rate {training_options.get('learning_rate')}, "
                 f"warmup {training_options.get('warmup_steps')}, epochs budget {training_options.get('epochs')}, "
                 f"max steps {training_options.get('max_steps')}, weight decay {training_options.get('weight_decay')}, "
                 f"clip {training_options.get('clip_grad')}")
    if paths["config"]:
        config = load_json(paths["config"])
        lines.append(f"- model: {config.get('n_layers')} layers x {config.get('d_model')} model dim, "
                     f"{config.get('n_heads')} heads, chunk {config.get('chunk_size')}, window "
                     f"{config.get('window_size')}, memory {config.get('memory_capacity')} chunks, top-k "
                     f"{config.get('memory_top_k')}, backend {config.get('memory_backend')}, vocab "
                     f"{config.get('vocab_size')}")
    records = read_log(paths["log"])
    # The final "finished" event has a step but no per-update metrics.
    steps = [record for record in records if record.get("step")]
    updates = [record for record in records if record.get("cursor")]
    if steps:
        last = steps[-1]
        timed = [record for record in records if record.get("tokens_per_second")]
        throughput = statistics.median(record["tokens_per_second"] for record in timed[-20:]) if timed else None
        lines.append(f"- completed {last['step']} updates, {last.get('tokens', 0):,} logged prediction targets"
                     + (f", median throughput {throughput:,.0f} targets/s" if throughput else ""))
        if updates:
            cursor = updates[-1].get("cursor", {})
            lines.append(f"- data cursor at the end: epoch {cursor.get('epoch')}, group {cursor.get('group')}, "
                         f"chunk {cursor.get('chunk')}; chunk memories in the active lanes: "
                         f"{updates[-1].get('memory_entries')}")
    if paths["plan"]:
        plan_data = load_json(paths["plan"])
        lines.append(f"- plan: {plan_data.get('planned_steps')} steps "
                     f"({plan_data.get('planned_targets_upper_bound', 0):,} targets upper bound, "
                     f"{plan_data.get('epochs_equivalent', 0):.2f} epochs over "
                     f"{plan_data.get('train_targets', 0):,} train targets); full pass = "
                     f"{plan_data.get('full_pass_steps')} steps")
        for note in plan_data.get("notes", []):
            lines.append(f"  - note: {note}")
    if paths["estimate"] and paths["estimate"].is_file():
        est = load_json(paths["estimate"])
        if est.get("median_tokens_per_second"):
            lines.append(f"- pilot throughput: {est['median_tokens_per_second']:,.0f} targets/s, "
                         f"{est['seconds_per_update']:.1f}s per update"
                         + (f"; projected {est['planned_hours']:.1f}h for {est.get('planned_steps')} planned steps"
                            if est.get("planned_hours") else ""))
    lines.append("")
    curve = [record for record in steps if "val_loss" in record]
    if curve:
        lines += ["## Validation", "",
                  "| step | val loss | val perplexity | val targets |", "|---:|---:|---:|---:|"]
        for record in curve:
            lines.append(f"| {record['step']} | {record['val_loss']:.4f} | {record['val_perplexity']:.2f} | "
                         f"{record.get('val_tokens', 0):,} |")
        best = min(curve, key=lambda record: record["val_loss"])
        lines += ["", f"Best logged validation loss {best['val_loss']:.4f} at step {best['step']} "
                      f"(perplexity {best['val_perplexity']:.2f}). Perplexity is tokenizer-dependent: never compare it "
                      "across vocabularies, and never treat it as a writing-quality score.", ""]
    for path in evaluations:
        record = load_json(path)
        split = record.get("split")
        loss = record.get(f"{split}_loss", record.get("val_loss"))
        perplexity = record.get(f"{split}_perplexity", record.get("val_perplexity"))
        tokens = record.get(f"{split}_tokens", record.get("val_tokens"))
        lines.append(f"- `{path.name}`: split {split} at checkpoint step {record.get('checkpoint_step')} — "
                     f"loss {loss:.4f}, perplexity {perplexity:.2f}, {tokens:,} targets"
                     if loss is not None else f"- `{path.name}`: no loss recorded")
    if evaluations:
        lines.append("")
    if paths["commands"]:
        lines += ["## Commands", "", "```bash", paths["commands"].read_text(encoding="utf-8").strip(), "```", ""]
    if samples_paths:
        lines += ["## Generation samples", ""]
        for path in samples_paths:
            samples = load_json(path)
            aggregate = samples.get("aggregate", {})
            if aggregate.get("continuation_loss") is None:
                lines.append(f"- `{path}`: {aggregate.get('documents')} documents, no scored continuation tokens")
                continue
            lines.append(f"- `{path}`: {aggregate.get('documents')} documents from split `{samples.get('split')}` at "
                         f"checkpoint step {samples.get('checkpoint_step')}; continuation CE "
                         f"{aggregate['continuation_loss']:.4f} "
                         f"(perplexity {aggregate['continuation_perplexity']:.2f}) over "
                         f"{aggregate['continuation_tokens']:,} held-out tokens"
                         + (f"; the same tokens score {aggregate['memory_off_loss']:.4f} with the retrieval tier "
                            "disabled" if aggregate.get("memory_off_loss") else ""))
        lines.append("")
        for path in samples_paths:
            markdown = find(path.parent, "samples.md")
            samples = load_json(path)
            if markdown:
                lines += [f"### In full: {samples.get('split')} split at checkpoint step "
                          f"{samples.get('checkpoint_step')}", "",
                          strip_heading(markdown.read_text(encoding="utf-8")), ""]
    lines += ["## How to read this", "",
              "- These numbers measure next-token prediction on held-out books, not narrative coherence. Read the "
              "generated text; a low perplexity with drifting text still means drift.",
              "- Continuation CE with the retrieval tier disabled isolates the effect of stored chunk memories "
              "**for this implementation on this prompt**; it is not a retrieval-recall benchmark.",
              "- A full PG-19-scale conclusion needs a fixed budget, a fixed checkpoint-selection rule and full-split "
              "evaluation on the official validation books. See `griffin_memory/PG19_RUNBOOK.md`.",
              "- Generated and reference text may be covered by third-party rights; do not redistribute it.",
              ""]
    report = "\n".join(lines)
    out = Path(args.out) if args.out else run_dir / "run_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(json.dumps({"event": "wrote", "report": str(out)}))
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    planner = sub.add_parser("plan", help="Steps and targets implied by a corpus and batch shape")
    planner.add_argument("--data", required=True)
    planner.add_argument("--config", default="griffin_memory/configs/small.json")
    planner.add_argument("--batch-size", type=int, default=1)
    planner.add_argument("--accumulation-steps", type=int, default=16)
    planner.add_argument("--epochs", type=int, default=3)
    planner.add_argument("--max-steps", type=int)
    planner.add_argument("--token-budget", type=int, help="Planned prediction targets; converts to steps")
    planner.add_argument("--out")
    planner.set_defaults(function=plan)

    estimator = sub.add_parser("estimate", help="Extrapolate wall-clock time from a training log")
    estimator.add_argument("--log", required=True)
    estimator.add_argument("--planned-steps", type=int)
    estimator.add_argument("--out")
    estimator.set_defaults(function=estimate)

    builder = sub.add_parser("build", help="Assemble run_report.md from a run directory")
    builder.add_argument("--run-dir", required=True)
    builder.add_argument("--title")
    builder.add_argument("--data", help="Prepared corpus directory, when it is not inside the run directory")
    builder.add_argument("--samples", action="append", help="samples.json; repeatable")
    builder.add_argument("--eval", dest="eval_", action="append", help="evaluation JSON; repeatable")
    builder.add_argument("--plan", help="plan.json override")
    builder.add_argument("--estimate", help="estimate.json override")
    builder.add_argument("--run-json", help="run.json override")
    builder.add_argument("--config", help="config.json override")
    builder.add_argument("--log", help="train.jsonl override")
    builder.add_argument("--commands", help="commands log override")
    builder.add_argument("--out")
    builder.set_defaults(function=build)
    return p


def main():
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
