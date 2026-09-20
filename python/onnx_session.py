"""CPU session settings shared by export verification and deployed inference."""

GRAPH_OPTIMIZATION_LEVELS = ("all", "basic", "disabled")


def cpu_session_options(*, graph_optimization_level="all", num_threads=0):
    import onnxruntime as ort
    if graph_optimization_level not in GRAPH_OPTIMIZATION_LEVELS:
        raise ValueError("ONNX Runtime graph_optimization_level must be all, basic or disabled")
    if type(num_threads) is not int or not 0 <= num_threads <= 2147483647:
        raise ValueError("ONNX Runtime num_threads must be a nonnegative int")
    options = ort.SessionOptions()
    options.intra_op_num_threads = num_threads
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = {
        "all": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        "basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "disabled": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
    }[graph_optimization_level]
    return options


def deployment_session_settings(config):
    settings = config.get("onnxruntime", {})
    if not isinstance(settings, dict):
        raise ValueError("onnxruntime must be an object")
    if config.get("schema_version") == 6 and (
            "graph_optimization_level" not in settings or "num_threads" not in config):
        raise ValueError("schema 6 requires the verified ONNX Runtime settings")
    level = settings.get("graph_optimization_level", "all")
    threads = config.get("num_threads", 0)
    if not isinstance(level, str) or level not in GRAPH_OPTIMIZATION_LEVELS:
        raise ValueError("Invalid ONNX Runtime graph_optimization_level")
    if type(threads) is not int or not 0 <= threads <= 2147483647:
        raise ValueError("Invalid ONNX Runtime num_threads")
    return {"graph_optimization_level": level, "num_threads": threads}
