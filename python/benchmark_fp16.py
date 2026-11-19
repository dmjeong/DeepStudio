"""Prepare an FP16 ONNX variant for a local CPU precision comparison."""

from graphlib import TopologicalSorter
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def _sort_graph_nodes(graph):
    nodes = []
    casts = {}
    for node in graph.node:
        # ORT's FP16 converter can append the very same Cast once for each
        # blocked consumer (e.g. horizontal and vertical CumSum). Keep one
        # producer, leaving any conflicting definitions for the checker.
        if node.op_type == "Cast" and not node.domain and len(node.output) == 1:
            output = node.output[0]
            if output in casts and node == casts[output]:
                continue
            casts[output] = node
        nodes.append(node)
    producers = {name: i for i, node in enumerate(nodes) for name in node.output if name}
    order = TopologicalSorter({
        i: [producers[name] for name in node.input if name in producers]
        for i, node in enumerate(nodes)
    })
    sorted_nodes = [nodes[i] for i in order.static_order()]
    graph.ClearField("node")
    graph.node.extend(sorted_nodes)


def convert_benchmark_fp16(fp32_path, output_dir) -> Path:
    """Create a temporary FP16 model and JSON while preserving FP32 model I/O."""
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    fp32_path = Path(fp32_path)
    config = json.loads(fp32_path.read_text(encoding="utf-8"))
    export = config.get("export", {})
    if (not isinstance(export, dict) or export.get("precision", "float32") != "float32"
            or export.get("compute_precision", "float32") != "float32" or config.get("encryption")):
        raise ValueError("FP16 비교 모델을 만들려면 암호화하지 않은 FP32 ONNX 설정 JSON이 필요합니다.")

    source = fp32_path.parent / config["model_path"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "benchmark-fp16.onnx"
    manifest_path = output_dir / "benchmark-fp16.json"
    if any(path.exists() or path.is_symlink() for path in (model_path, manifest_path)):
        raise FileExistsError("FP16 비교용 출력 위치에 기존 파일이 있습니다.")

    try:
        model = onnx.load(source)
        inputs = [item for item in model.graph.input if item.name == config["input_name"]]
        outputs = list(model.graph.output)
        output_names = config.get("output_names") or [config["output_name"]]
        float32 = onnx.TensorProto.FLOAT
        if (len(inputs) != 1 or [item.name for item in outputs] != output_names
                or any(item.type.tensor_type.elem_type != float32 for item in (*inputs, *outputs))):
            raise ValueError("ONNX 입력·출력 이름 또는 FP32 타입이 설정 JSON과 다릅니다.")
        fp16 = onnx.TensorProto.FLOAT16
        if (any(tensor.data_type == fp16 for tensor in model.graph.initializer)
                or any(node.op_type in {"QuantizeLinear", "DequantizeLinear"} for node in model.graph.node)):
            raise ValueError("원본 그래프에 이미 FP16 또는 양자화 연산이 있습니다.")

        architecture = str(config.get("architecture", ""))
        if architecture.startswith("dinov2_") or architecture == "mask2former_swin_small":
            # Fold DINOv2 position/attention constants and Mask2Former position
            # grids before the FP16 converter inserts casts for their consumers.
            import onnxruntime as ort
            from onnx_session import cpu_session_options

            del inputs, outputs, model
            with TemporaryDirectory(dir=output_dir) as temp:
                folded = Path(temp) / "folded.onnx"
                options = cpu_session_options(graph_optimization_level="basic", num_threads=1)
                options.optimized_model_filepath = str(folded)
                ort.InferenceSession(str(source), options, providers=["CPUExecutionProvider"])
                model = onnx.load(folded)
        model = convert_float_to_float16(model, keep_io_types=True)
        # The converter appends the I/O and blocked-operator Cast nodes.
        _sort_graph_nodes(model.graph)
        onnx.checker.check_model(model)
        if not any(tensor.data_type == fp16 for tensor in model.graph.initializer):
            raise ValueError("FP16으로 바뀐 가중치가 없습니다.")
    except Exception as error:
        raise ValueError(f"FP16 비교 모델 변환 실패: {error}") from error

    config["model_path"] = model_path.name
    config["verification"] = "benchmark_only"
    config["export"] = {key: export[key] for key in ("opset", "dynamic_batch") if key in export}
    config["export"].update(precision="float16", io_precision="float32")
    onnx.save(model, model_path)
    manifest_path.write_text(json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return manifest_path
