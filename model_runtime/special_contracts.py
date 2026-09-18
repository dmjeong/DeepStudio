"""Manifest-level contracts for model families with multiple ONNX graphs.

These checks do not claim that an upstream checkpoint is present.  They make
an added Re-DETR or SAM2 model explicit enough for a Docker worker and prevent
the generic one-output runtime from silently opening a special graph.
"""

from __future__ import annotations

from collections.abc import Mapping


class SpecialContractError(ValueError):
    """A special model manifest is missing a required deployment contract."""


RE_DETR_VARIANTS = frozenset({"Small", "Medium", "Large"})
SAM2_VARIANTS = frozenset({"Hiera Tiny", "Hiera Small", "Hiera Base+", "Hiera Large"})


def _require_mapping(value, name: str) -> Mapping:
    if not isinstance(value, Mapping):
        raise SpecialContractError(f"{name} must be an object")
    return value


def _require_text(value, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SpecialContractError(f"{name} must be a non-empty string")
    return value


def validate_re_detr_manifest(manifest: Mapping) -> None:
    if manifest.get("family") != "Re-DETR v4" or manifest.get("task") != "detect":
        raise SpecialContractError("Re-DETR manifest must use family Re-DETR v4 and detect task")
    if manifest.get("variant") not in RE_DETR_VARIANTS:
        raise SpecialContractError("Re-DETR variant must be Small, Medium or Large")
    if "onnx" not in manifest.get("runtimes", ()):
        raise SpecialContractError("Re-DETR manifest requires an ONNX runtime")
    contract = _require_mapping(manifest.get("contracts"), "contracts")
    onnx = _require_mapping(contract.get("onnx"), "contracts.onnx")
    for key in ("file", "input_name", "boxes_name", "logits_name", "boxes_format"):
        _require_text(onnx.get(key), f"contracts.onnx.{key}")
    if onnx["boxes_format"] not in {"normalized_cxcywh", "normalized_xyxy"}:
        raise SpecialContractError("Re-DETR boxes_format is unsupported")
    if onnx.get("score_activation", "sigmoid") not in {"sigmoid", "softmax"}:
        raise SpecialContractError("Re-DETR score_activation is unsupported")


def validate_sam2_manifest(manifest: Mapping) -> None:
    if manifest.get("family") != "SAM2" or manifest.get("task") != "segment":
        raise SpecialContractError("SAM2 manifest must use family SAM2 and segment task")
    if manifest.get("variant") not in SAM2_VARIANTS:
        raise SpecialContractError("SAM2 variant is unsupported")
    if not {"onnx", "container"}.issubset(set(manifest.get("runtimes", ()) )):
        raise SpecialContractError("SAM2 requires ONNX and container runtimes")
    contract = _require_mapping(manifest.get("contracts"), "contracts")
    graphs = _require_mapping(contract.get("graphs"), "contracts.graphs")
    for graph_name in ("encoder", "decoder"):
        graph = _require_mapping(graphs.get(graph_name), f"contracts.graphs.{graph_name}")
        _require_text(graph.get("file"), f"contracts.graphs.{graph_name}.file")
        outputs = graph.get("outputs")
        if not isinstance(outputs, list) or not outputs or any(not isinstance(name, str) or not name for name in outputs):
            raise SpecialContractError(f"contracts.graphs.{graph_name}.outputs must be a string list")
        inputs = _require_mapping(graph.get("inputs"), f"contracts.graphs.{graph_name}.inputs")
        if any(not isinstance(name, str) or not name for name in inputs.values()):
            raise SpecialContractError(f"contracts.graphs.{graph_name}.inputs must map to tensor names")
        required_inputs = {"image"} if graph_name == "encoder" else {
            "image_embeddings", "point_coords", "point_labels", "mask_input",
            "has_mask_input", "orig_im_size",
        }
        if not required_inputs.issubset(inputs):
            raise SpecialContractError(f"contracts.graphs.{graph_name}.inputs is incomplete")
    prompts = contract.get("prompt_types")
    if not isinstance(prompts, list) or not prompts or any(prompt not in {"point", "box", "mask"} for prompt in prompts):
        raise SpecialContractError("SAM2 prompt_types must contain point, box or mask")
    if not isinstance(contract.get("video_state"), bool):
        raise SpecialContractError("SAM2 video_state must be boolean")
    automatic = contract.get("automatic_mask")
    if automatic is not None:
        automatic = _require_mapping(automatic, "contracts.automatic_mask")
        if automatic.get("mode") != "positive_point_grid_union":
            raise SpecialContractError("SAM2 automatic_mask mode is unsupported")
        max_grid = automatic.get("max_grid")
        if isinstance(max_grid, bool) or not isinstance(max_grid, int) or not 1 <= max_grid <= 32:
            raise SpecialContractError("SAM2 automatic_mask max_grid must be between 1 and 32")


def validate_special_manifest(manifest: Mapping) -> None:
    """Validate only families that have a special multi-graph contract."""
    family = manifest.get("family")
    if family == "Re-DETR v4":
        validate_re_detr_manifest(manifest)
    elif family == "SAM2":
        validate_sam2_manifest(manifest)


def validate_container_entrypoint(manifest: Mapping) -> None:
    """Validate the optional Docker plugin factory without importing it."""
    entrypoint = manifest.get("worker_entrypoint")
    if entrypoint is None:
        return
    if not isinstance(entrypoint, str) or entrypoint.count(":") != 1:
        raise SpecialContractError("worker_entrypoint must use module:factory")
    module, factory = entrypoint.split(":", 1)
    if (not module or not factory or any(part in module for part in ("/", "\\", "..")) or
            any(char.isspace() or char == "\x00" for char in entrypoint)):
        raise SpecialContractError("worker_entrypoint contains an unsafe module or factory")


def validate_container_image_asset(manifest: Mapping, asset_names) -> None:
    """Validate an optional offline Docker image archive declared by a pack.

    A digest-pinned image may be preloaded from a pack-local ``docker save``
    archive.  The archive is optional for development packs that use an image
    already loaded into the application-owned engine, but when declared it
    must be covered by the pack's checksum set.
    """
    archive = manifest.get("container_image_archive")
    if archive is None:
        return
    if (not isinstance(archive, str) or not archive.strip() or archive.startswith("/") or
            ":" in archive or "\\" in archive or "//" in archive or ".." in archive.split("/")):
        raise SpecialContractError("container_image_archive must be a safe relative path")
    names = set(asset_names)
    if archive not in names:
        raise SpecialContractError(f"container image archive is missing from pack: {archive}")


def special_asset_paths(manifest: Mapping) -> tuple[str, ...]:
    """Return graph files declared by a special model contract."""
    family = manifest.get("family")
    if family == "SAM2":
        contracts = manifest.get("contracts")
        graphs = contracts.get("graphs") if isinstance(contracts, Mapping) else None
        if isinstance(graphs, Mapping):
            return tuple(graph.get("file") for graph in graphs.values()
                         if isinstance(graph, Mapping) and isinstance(graph.get("file"), str))
    if family == "Re-DETR v4":
        contracts = manifest.get("contracts")
        onnx = contracts.get("onnx") if isinstance(contracts, Mapping) else None
        if isinstance(onnx, Mapping):
            if isinstance(onnx.get("file"), str):
                return (onnx["file"],) + tuple(path for path in onnx.get("files", ()) if isinstance(path, str))
            declared = onnx.get("files", ())
            if isinstance(declared, list):
                return tuple(path for path in declared if isinstance(path, str))
    return ()


def validate_special_assets(manifest: Mapping, asset_names) -> None:
    """Ensure every graph explicitly declared by a special pack is included."""
    validate_special_manifest(manifest)
    names = set(asset_names)
    for path in special_asset_paths(manifest):
        if not path or path.startswith("/") or "\\" in path or ".." in path.split("/") or path not in names:
            raise SpecialContractError(f"special model graph is missing from pack: {path}")
