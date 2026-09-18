"""Tune the actual C++17 ONNX Runtime binary on the target CPU (standard library only)."""
import argparse
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess


def cpu_name():
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def integer(text, minimum=1):
    value = int(text)
    if value < minimum:
        raise argparse.ArgumentTypeError(f"Expected an integer >= {minimum}")
    return value


def without_samples(value):
    if isinstance(value, dict):
        return {k: without_samples(v) for k, v in value.items() if k != "samples_ms"}
    if isinstance(value, list):
        return [without_samples(v) for v in value]
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    image = parser.add_mutually_exclusive_group(required=True)
    image.add_argument("--image", type=Path)
    image.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; never overwrite prior measurements")
    parser.add_argument("--threads", type=integer, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--block-bases", type=lambda s: integer(s, 0), nargs="+", default=[0, 4])
    parser.add_argument("--spinning", nargs="+", choices=["default", "on", "off"], default=["default", "off"])
    parser.add_argument("--rounds", type=integer, default=2)
    parser.add_argument("--warmup", type=integer, default=30)
    parser.add_argument("--runs", type=integer, default=200)
    parser.add_argument("--direct", action="store_true", help="Default measures background Submit through future completion")
    parser.add_argument("--cpu-label", default=cpu_name())
    args = parser.parse_args(argv)
    executable, config_path, output = args.executable.resolve(), args.config.resolve(), args.output.resolve()
    if not executable.is_file():
        parser.error("C++ benchmark executable not found")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model = (config_path.parent / config["model_path"]).resolve()
    if not model.is_file():
        parser.error("ONNX model not found")
    if output.exists():
        parser.error("Output directory already exists; choose a new directory")
    output.mkdir(parents=True)
    # All generated manifests resolve the same original model. Source files are untouched.
    try:
        config["model_path"] = os.path.relpath(model, output)
    except ValueError:  # Windows paths on different drives.
        config["model_path"] = str(model)
    config["runtime"] = "onnxruntime"
    candidates = []
    for threads, block, spin in sorted(set(itertools.product(args.threads, args.block_bases, args.spinning))):
        candidate = dict(config, num_threads=threads, onnxruntime={"dynamic_block_base": block})
        if spin != "default":
            candidate["onnxruntime"]["allow_spinning"] = spin == "on"
        name = f"threads-{threads}-block-{block}-spin-{spin}"
        manifest = output / f"{name}.json"
        manifest.write_text(json.dumps(candidate, indent=2, ensure_ascii=False), encoding="utf-8")
        candidates.append({"name": name, "config": manifest.name, "threads": threads,
                           "options": {"allow_spinning": {"default": -1, "on": 1, "off": 0}[spin],
                                       "dynamic_block_base": block}, "trials": []})
    reference = None
    for round_index in range(args.rounds):
        order = list(candidates)
        random.Random(42 + round_index).shuffle(order)
        for candidate in order:
            result_path = output / f"{candidate['name']}-round-{round_index}.json"
            command = [str(executable), "--config", str(output / candidate["config"]), "--runtime", "onnxruntime",
                       "--threads", str(candidate["threads"]), "--warmup", str(args.warmup), "--runs", str(args.runs),
                       "--cpu-label", args.cpu_label, "--output", str(result_path)]
            command += ["--image", str(args.image.resolve())] if args.image else ["--synthetic"]
            if not args.direct:
                command.append("--background")
            process = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                                     timeout=max(120, args.runs * 2))
            if process.returncode:
                raise RuntimeError(f"{candidate['name']} failed: {process.stderr}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if "onnxruntime_options" not in result:
                raise RuntimeError("Rebuild the C++ benchmark with ONNX Runtime tuning support first")
            if result["onnxruntime_options"] != candidate["options"] or result["threads"] != candidate["threads"]:
                raise RuntimeError("C++ executable did not apply the requested settings")
            prediction = (result["last_class_id"], result["last_probabilities"])
            if reference is None:
                reference = prediction
            if prediction[0] != reference[0] or len(prediction[1]) != len(reference[1]) or any(
                not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5) for a, b in zip(prediction[1], reference[1])
            ):
                raise RuntimeError("Prediction changed across settings; review correctness before selecting one")
            candidate["trials"].append({"file": result_path.name, "result": without_samples(result)})
            print(f"{candidate['name']} round={round_index + 1} p95={result['wall']['p95_ms']:.3f} ms", flush=True)
    for candidate in candidates:
        candidate["worst_round_p95_ms"] = max(t["result"]["wall"]["p95_ms"] for t in candidate["trials"])
    candidates.sort(key=lambda c: c["worst_round_p95_ms"])
    selected = candidates[0]
    (output / "recommended.json").write_text((output / selected["config"]).read_text(encoding="utf-8"), encoding="utf-8")
    report = {"cpu_label": args.cpu_label, "os": platform.platform(), "background": not args.direct,
              "selection": "lowest worst-round wall p95; one request at a time, setup/decode excluded",
              "selected": selected["name"], "rounds": args.rounds, "runs_per_round": args.runs,
              "warning": "Recheck with at least 1000 calls and production load; no hard deadline guarantee or accuracy evaluation",
              "candidates": candidates}
    (output / "summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Measured candidate: {selected['name']}; config: {output / 'recommended.json'}")
    return report


if __name__ == "__main__":
    main()
