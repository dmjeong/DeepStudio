"""Bundle the plugins loaded dynamically by OpenVINO and their notices."""
from PyInstaller.utils.hooks import collect_dynamic_libs, copy_metadata

# Keep openvino/libs so the Python bindings and the native ONNX frontend can
# find the CPU plugin and oneTBB without a system OpenVINO installation.
binaries = collect_dynamic_libs("openvino", search_patterns=["*.dll"])
datas = copy_metadata("openvino")
