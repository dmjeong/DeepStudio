"""Preserve the source-frame ONNX interface around LibreYOLO's native canvas."""
import io

import onnx
from onnx import helper, numpy_helper
import numpy as np
import torch

from native_detection import source_frame_outputs
from upstream_original_size import detector_size, input_hw


class _SourceFrameOutputs(torch.nn.Module):
    def __init__(self, family, region):
        super().__init__()
        self.family = family
        self.region = region

    def forward(self, *outputs):
        return tuple(source_frame_outputs(self.family, self.region, outputs))


def add_source_frame_interface(path, family, region, num_classes, opset):
    """Keep the verified native exporter; add only border padding and box decoding."""
    width, height = region
    canvas_h, canvas_w = input_hw(detector_size(family, height, width))
    if (height, width) == (canvas_h, canvas_w):
        return
    model = onnx.load(str(path))
    graph = model.graph
    image = graph.input[0]
    pad_name, pixel_name = "studio_source_pads", "studio_source_pad_value"
    graph.initializer.extend([
        numpy_helper.from_array(np.array([0, 0, 0, 0, 0, 0, canvas_h - height, canvas_w - width], np.int64), pad_name),
        numpy_helper.from_array(np.array(114 / 255 if family == "yolo9" else 0, np.float32), pixel_name),
    ])
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == image.name:
                node.input[index] = "studio_canvas_image"
    graph.node.insert(0, helper.make_node("Pad", [image.name, pad_name, pixel_name], ["studio_canvas_image"]))
    image.type.tensor_type.shape.dim[2].dim_value = height
    image.type.tensor_type.shape.dim[3].dim_value = width

    # Trace the same coordinate transform used during validation before top-k/NMS.
    # Its slicing references fixed axes only, independent of batch/anchor counts.
    inputs = ([torch.zeros(1, 4 + num_classes, 8)] if family == "yolo9" else
              [torch.zeros(1, 8, num_classes), torch.zeros(1, 8, 4)])
    input_names = [f"raw_{i}" for i in range(len(inputs))]
    output_names = [f"frame_{i}" for i in range(len(inputs))]
    buffer = io.BytesIO()
    torch.onnx.export(_SourceFrameOutputs(family, tuple(region)), tuple(inputs), buffer,
                      input_names=input_names, output_names=output_names, opset_version=opset,
                      dynamic_axes={name: {0: "batch", (2 if family == "yolo9" else 1): "candidates"}
                                    for name in input_names + output_names},
                      dynamo=False)
    tail = onnx.compose.add_prefix(onnx.load_from_string(buffer.getvalue()), "studio_coordinates_").graph
    sources = {value.name: source.name for value, source in zip(tail.input, graph.output)}
    for node in tail.node:
        for index, name in enumerate(node.input):
            if name in sources:
                node.input[index] = sources[name]
    graph.initializer.extend(tail.initializer)
    graph.node.extend(tail.node)
    for output, mapped in zip(graph.output, tail.output):
        output.name = mapped.name
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
