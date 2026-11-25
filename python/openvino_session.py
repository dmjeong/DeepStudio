"""OpenVINO CPU settings shared by desktop benchmarks and classification."""


def compile_cpu_model(core, model, num_threads):
    if type(num_threads) is not int or num_threads < 0:
        raise ValueError("스레드 수는 0 이상의 정수 필요")
    # Keep floating-point kernels at f32. Explicit quantization in the graph
    # can still use integer kernels where the CPU plugin supports them.
    options = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 1,
               "INFERENCE_PRECISION_HINT": "f32"}
    if num_threads:
        options["INFERENCE_NUM_THREADS"] = num_threads
    compiled = core.compile_model(model, "CPU", options)
    settings = {key: str(compiled.get_property(key)) for key in options}
    settings["INFERENCE_NUM_THREADS"] = int(compiled.get_property("INFERENCE_NUM_THREADS"))
    settings["device"] = core.get_property("CPU", "FULL_DEVICE_NAME")
    return compiled, settings
