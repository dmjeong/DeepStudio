"""Real masked-defect LoRA training for 9-channel SD/SDXL inpainting models.

Optional runtime, loaded only inside the shared worker process. No rule fallback.
"""
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image
from core.datagen_store import DataGenStore, checkpoint_decision, image8, mask_image, save_png, uid
from webapp.storage import digest, read_json, write_json


def crop_pair(image, mask, resolution):
    """Square context without anisotropic resizing, with reversible original coordinates."""
    bounds = mask.getbbox()
    if not bounds:
        raise ValueError("생성 또는 학습 마스크가 비었습니다")
    x0, y0, x1, y1 = bounds
    side = max(64, 2 * max(x1 - x0, y1 - y0))
    left, top = (x0 + x1 - side) // 2, (y0 + y1 - side) // 2
    box = (left, top, left + side, top + side)
    pixels = np.array(image.convert("RGB"))
    padding = ((max(0, -top), max(0, box[3] - image.height)),
               (max(0, -left), max(0, box[2] - image.width)), (0, 0))
    padded = np.pad(pixels, padding, mode="edge")
    sx, sy = max(left, 0), max(top, 0)
    patch = Image.fromarray(padded[sy:sy + side, sx:sx + side]).resize((resolution, resolution), Image.Resampling.LANCZOS)
    cropped_mask = mask.crop(box).resize((resolution, resolution), Image.Resampling.NEAREST)
    return patch, cropped_mask, box


def compose(original, generated, requested, allowed, box):
    request = np.array(requested) > 127
    permit = np.array(allowed) > 127
    if not request.any() or np.any(request & ~permit):
        raise ValueError("생성 요청 마스크는 비어 있지 않고 허용 영역 안에 있어야 합니다")
    x0, y0, x1, y1 = box
    patch = generated.convert(original.mode).resize((x1 - x0, y1 - y0), Image.Resampling.LANCZOS)
    canvas = original.copy()
    canvas.paste(patch, (x0, y0))
    source, output = np.array(original), np.array(canvas)
    output[~request] = source[~request]
    changed = np.any(output != source, axis=2) if output.ndim == 3 else output != source
    if np.any(changed & ~permit) or not changed.any():
        raise ValueError("원본 보존 실패 또는 실제 변경이 없는 생성 결과")
    return Image.fromarray(output), Image.fromarray(changed.astype("uint8") * 255)


def config(raw):
    defaults = {"base_model": "", "resolution": 512, "steps": 500, "validate_every": 50,
                "learning_rate": 0.0001, "rank": 8, "seed": 42, "precision": "fp32",
                "background_weight": 0.1, "min_delta": 0.0, "patience": 0,
                "prompt": "", "validation_seeds": [1001, 2002, 3003]}
    if set(raw) - set(defaults):
        raise ValueError("지원하지 않는 학습 설정")
    result = {**defaults, **raw}
    for key, lo, hi in (("resolution", 256, 1024), ("steps", 1, 100000), ("validate_every", 1, 10000),
                        ("rank", 1, 128), ("seed", 0, 4294967295), ("patience", 0, 1000)):
        if type(result[key]) is not int or not lo <= result[key] <= hi:
            raise ValueError(f"{key} 설정 범위 오류")
    if result["resolution"] % 64 or result["precision"] not in {"fp32", "fp16", "bf16"}:
        raise ValueError("해상도는 64의 배수, 정밀도는 fp32/fp16/bf16이어야 합니다")
    for key, lo, hi in (("learning_rate", 0, .1), ("background_weight", 0, 10), ("min_delta", 0, 100)):
        if not isinstance(result[key], (float, int)) or not math.isfinite(result[key]) or not lo <= result[key] <= hi:
            raise ValueError(f"{key} 설정 범위 오류")
    if result["learning_rate"] <= 0 or not str(result["prompt"]).strip():
        raise ValueError("양수 학습률과 학습할 외관 설명을 입력하세요")
    if result["validation_seeds"] != defaults["validation_seeds"]:
        raise ValueError("이 데이터 버전의 고정 검증 시드를 유지해야 합니다")
    return result


def runtime():
    try:
        import torch
        import diffusers
        import peft
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError("Data Gen 전용 환경이 필요합니다. docs/DATA_GEN.md의 설치 절차와 DEEP_STUDIO_DATAGEN_PYTHON을 확인하세요") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("실제 불량 학습과 AI 생성에는 CUDA GPU가 필요합니다. 데이터 준비와 검수는 CPU에서 가능합니다")
    return torch, diffusers, peft, load_file, save_file


def base_fingerprint(path):
    root = Path(path)
    if not root.is_absolute() or not (root / "model_index.json").is_file():
        raise ValueError("다운로드한 SD 또는 SDXL Inpainting 모델 폴더의 절대 경로를 지정하세요")
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in {".json", ".txt", ".safetensors", ".model"})
    values = [(str(p.relative_to(root)), digest(p)) for p in files]
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()


def load_pipeline(cfg, torch, diffusers):
    family = read_json(Path(cfg["base_model"]) / "model_index.json").get("_class_name")
    classes = {"StableDiffusionInpaintPipeline": diffusers.StableDiffusionInpaintPipeline,
               "StableDiffusionXLInpaintPipeline": diffusers.StableDiffusionXLInpaintPipeline}
    if family not in classes:
        raise ValueError("SD 또는 SDXL 전용 Inpainting 가중치를 선택하세요")
    pipe = classes[family].from_pretrained(cfg["base_model"], local_files_only=True, use_safetensors=True,
                                          torch_dtype=torch.float32).to("cuda")
    if pipe.unet.config.in_channels != 9 or pipe.unet.config.out_channels != 4:
        raise ValueError("9채널 인페인팅 UNet과 4채널 잠재 공간이 필요합니다")
    for component in (pipe.unet, pipe.vae, pipe.text_encoder, getattr(pipe, "text_encoder_2", None)):
        if component is not None:
            component.requires_grad_(False)
            component.eval()
    if cfg["precision"] == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("이 GPU는 BF16을 지원하지 않습니다. FP32 또는 FP16을 선택하세요")
    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()
    return pipe


def add_adapter(pipe, cfg, peft):
    pipe.unet.add_adapter(peft.LoraConfig(r=cfg["rank"], lora_alpha=cfg["rank"], init_lora_weights="gaussian",
                                        target_modules=["to_q", "to_k", "to_v", "to_out.0"]))


def area_loss(prediction, target, mask, background_weight):
    errors = (prediction.float() - target.float()).square().mean(dim=1, keepdim=True)
    dimensions = (1, 2, 3)
    defect = ((errors * mask).sum(dim=dimensions) / mask.sum(dim=dimensions).clamp_min(1)).mean()
    background = ((errors * (1 - mask)).sum(dim=dimensions) / (1 - mask).sum(dim=dimensions).clamp_min(1)).mean()
    return defect + background_weight * background, defect, background


def train(context, payload, project):
    torch, diffusers, peft, load_file, save_file = runtime()
    store = DataGenStore(project)
    resume_id = payload.get("resume_model", "")
    if resume_id:
        model = store.record("models", resume_id)
        cfg = model["config"]
        dataset = store.record("datasets", model["dataset_id"])
        if not model.get("last"):
            raise ValueError("재개할 완료 체크포인트가 없습니다")
    else:
        cfg = config(payload.get("config", {}))
        dataset = store.record("datasets", payload["dataset_id"])
        model = {"id": uid(), "item_id": dataset["item_id"], "dataset_id": dataset["id"], "item": dataset["item"],
                 "config": cfg, "checkpoints": [], "best_val_loss": None, "best_quality": None, "last": None,
                 "status": "preparing", "quality_status": "not_validated", "created_at": time.time()}
    root = store.root / "models" / model["id"]
    root.mkdir(parents=True, exist_ok=True)
    context.emit("log_message", ["모델 파일 무결성과 실제 train/val 데이터를 확인합니다"])
    fingerprint = base_fingerprint(cfg["base_model"])
    if resume_id and model["base_sha256"] != fingerprint:
        raise ValueError("베이스 모델이 변경되어 학습을 재개할 수 없습니다")
    model["base_sha256"] = fingerprint
    model["environment"] = {name: importlib.metadata.version(name) for name in ("torch", "diffusers", "peft", "transformers")}
    dataset_root = store.root / "datasets" / dataset["id"]
    for row in dataset["images"]:
        for filename, key in (("image.png", "image_sha256"), ("mask.png", "mask_sha256")):
            if digest(dataset_root / row["id"] / filename) != row[key]:
                raise ValueError("고정 데이터 버전의 파일 무결성 오류")
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    pipe = load_pipeline(cfg, torch, diffusers)
    add_adapter(pipe, cfg, peft)
    pipe.unet.enable_gradient_checkpointing()
    parameters = [v for v in pipe.unet.parameters() if v.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    noise_scheduler = diffusers.DDPMScheduler.from_config(pipe.scheduler.config)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["precision"] == "fp16")
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[cfg["precision"]]
    def amp():
        return torch.autocast("cuda", dtype=dtype) if dtype != torch.float32 else nullcontext()
    random = torch.Generator(device="cuda").manual_seed(cfg["seed"])
    sampler = torch.Generator().manual_seed(cfg["seed"])
    train_rows = [v for v in dataset["images"] if v["split"] == "train"]
    val_rows = [v for v in dataset["images"] if v["split"] == "val"]
    if not train_rows or not val_rows:
        raise ValueError("실제 train과 val 데이터가 모두 필요합니다")
    with torch.no_grad():
        encoded = pipe.encode_prompt(cfg["prompt"], device="cuda", num_images_per_prompt=1, do_classifier_free_guidance=False)
        embedding = encoded[0]
        extra = {"text_embeds": encoded[2], "time_ids": torch.tensor([[cfg["resolution"], cfg["resolution"], 0, 0,
                  cfg["resolution"], cfg["resolution"]]], device="cuda", dtype=embedding.dtype)} if hasattr(pipe, "text_encoder_2") else None

    def forward(row, rng):
        image = image8(dataset_root / row["id"] / "image.png")
        mask = mask_image(dataset_root / row["id"] / "mask.png", image.size)
        patch, mask, _ = crop_pair(image, mask, cfg["resolution"])
        pixels = torch.from_numpy(np.array(patch).copy()).permute(2, 0, 1).unsqueeze(0).to("cuda", torch.float32) / 127.5 - 1
        binary = torch.from_numpy((np.array(mask) > 127).astype("float32")).unsqueeze(0).unsqueeze(0).to("cuda")
        with torch.no_grad():
            latent = pipe.vae.encode(pixels).latent_dist.mode() * pipe.vae.config.scaling_factor
            masked = pipe.vae.encode(pixels * (1 - binary)).latent_dist.mode() * pipe.vae.config.scaling_factor
            latent_mask = torch.nn.functional.interpolate(binary, size=latent.shape[-2:], mode="nearest")
            if not latent_mask.any():
                raise ValueError("잠재 공간에서 사라지는 작은 불량입니다. 마스크 주변 패치와 학습 해상도를 조정하세요")
            noise = torch.randn(latent.shape, generator=rng, device="cuda")
            timestep = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), generator=rng, device="cuda").long()
            noisy = noise_scheduler.add_noise(latent, noise, timestep)
            prediction_type = noise_scheduler.config.prediction_type
            if prediction_type == "epsilon":
                target = noise
            elif prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latent, noise, timestep)
            else:
                raise ValueError(f"지원하지 않는 scheduler 예측 대상: {prediction_type}")
        with amp():
            prediction = pipe.unet(torch.cat((noisy, latent_mask, masked), dim=1), timestep,
                                   encoder_hidden_states=embedding, added_cond_kwargs=extra).sample
            return area_loss(prediction, target, latent_mask, cfg["background_weight"])

    step, order, position = 0, [], 0
    if resume_id:
        checkpoint = root / "checkpoints" / model["last"]
        saved = read_json(checkpoint / "manifest.json")
        if digest(checkpoint / "adapter.safetensors") != saved["adapter_sha256"] or digest(checkpoint / "training.pt") != saved["training_sha256"]:
            raise ValueError("이어학습 체크포인트 무결성 오류")
        peft.set_peft_model_state_dict(pipe.unet, load_file(str(checkpoint / "adapter.safetensors")))
        state = torch.load(checkpoint / "training.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        random.set_state(state["noise_rng"])
        sampler.set_state(state["sampler_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        step, order, position = state["step"], state["order"], state["position"]
    model["status"] = "training"
    write_json(root / "manifest.json", model)
    last_train = None

    def checkpoint(validation=None):
        nonlocal model
        key = uid()
        temporary = root / "checkpoints" / f".{key}.partial"
        temporary.mkdir(parents=True)
        try:
            adapter = {k: v.detach().cpu().contiguous() for k, v in peft.get_peft_model_state_dict(pipe.unet).items()}
            save_file(adapter, str(temporary / "adapter.safetensors"))
            torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "noise_rng": random.get_state(), "sampler_rng": sampler.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all(), "step": step, "order": order, "position": position}, temporary / "training.pt")
            record = {"id": key, "step": step, "epoch": step / len(train_rows), "train_loss": last_train,
                      "val_loss": validation[0] if validation else None,
                      "val_defect_loss": validation[1] if validation else None,
                      "val_background_loss": validation[2] if validation else None,
                      "adapter_sha256": digest(temporary / "adapter.safetensors"), "training_sha256": digest(temporary / "training.pt"), "time": time.time()}
            write_json(temporary / "manifest.json", record)
            temporary.rename(root / "checkpoints" / key)
            model["checkpoints"].append(record)
            model["last"] = key
            model, stop = checkpoint_decision(model, record["val_loss"], key, cfg["min_delta"], cfg["patience"])
            write_json(root / "manifest.json", model)
            context.emit("datagen_checkpoint", [record])
            return stop
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    try:
        while step < cfg["steps"] and not context.cancelled():
            if position >= len(order):
                order = torch.randperm(len(train_rows), generator=sampler).tolist()
                position = 0
            pipe.unet.train()
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = forward(train_rows[order[position]], random)
            if not torch.isfinite(loss):
                raise ValueError("학습 손실이 NaN/Inf입니다. 기존 체크포인트를 보존합니다")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            position += 1
            last_train = float(loss.detach())
            context.emit("progress_updated", [step, cfg["steps"]])
            if step % cfg["validate_every"] == 0 or step == cfg["steps"]:
                pipe.unet.eval()
                values = []
                with torch.no_grad():
                    for row in val_rows:
                        for seed in cfg["validation_seeds"]:
                            if context.cancelled():
                                break
                            stable_seed = (int(row["pixel_sha256"][:8], 16) + seed) % 4294967296
                            values.append([float(v) for v in forward(row, torch.Generator(device="cuda").manual_seed(stable_seed))])
                        if context.cancelled():
                            break
                measured = np.mean(values, axis=0).tolist() if values and not context.cancelled() else None
                if measured and not all(math.isfinite(v) for v in measured):
                    context.emit("log_message", ["검증 손실 NaN/Inf: Best를 변경하지 않습니다"])
                    measured = None
                if checkpoint(measured):
                    context.emit("log_message", ["연속 미개선 기준에 따라 조기 종료했습니다"])
                    break
        if not model["checkpoints"] or model["checkpoints"][-1]["step"] != step:
            checkpoint()
        model["status"] = "cancelled" if context.cancelled() else "completed"
    except Exception:
        model["status"] = "failed"
        write_json(root / "manifest.json", model)
        raise
    finally:
        model["gpu"] = torch.cuda.get_device_name()
        model["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
        write_json(root / "manifest.json", model)
    return {"status": model["status"], "output": {"model_id": model["id"], "best_val_loss": model["best_val_loss"], "last": model["last"]}}


def generate(context, payload, project):
    torch, diffusers, peft, load_file, _ = runtime()
    store = DataGenStore(project)
    model = store.record("models", payload["model_id"])
    cfg = model["config"]
    key = payload.get("checkpoint") or model.get("best_quality") or model.get("best_val_loss")
    record = next((v for v in model["checkpoints"] if v["id"] == key), None)
    if record is None:
        raise ValueError("검증 손실이 측정된 모델 또는 명시적인 완료 체크포인트를 선택하세요")
    checkpoint = store.root / "models" / model["id"] / "checkpoints" / key / "adapter.safetensors"
    if digest(checkpoint) != record["adapter_sha256"] or base_fingerprint(cfg["base_model"]) != model["base_sha256"]:
        raise ValueError("모델 가중치 무결성 오류")
    row = next((v for v in store.state()["images"] if v["id"] == payload["image_id"]), None)
    if row is None or row["role"] != "normal" or row["item_id"] != model["item_id"]:
        raise ValueError("이 학습 항목에 연결된 실제 정상 이미지를 선택하세요")
    image = image8(store.file("images", row["id"]))
    requested = mask_image(payload["requested"], image.size)
    allowed = mask_image(payload.get("allowed") or payload["requested"], image.size)
    if np.any((np.array(requested) > 0) & (np.array(allowed) == 0)):
        raise ValueError("요청 영역이 허용 영역을 벗어났습니다")
    count, seed = payload.get("count", 1), payload.get("seed", 42)
    if type(count) is not int or not 1 <= count <= 100 or type(seed) is not int or not 0 <= seed < 4294967296:
        raise ValueError("생성 수는 1~100, 시드는 0~4294967295입니다")
    pipe = load_pipeline(cfg, torch, diffusers)
    add_adapter(pipe, cfg, peft)
    peft.set_peft_model_state_dict(pipe.unet, load_file(str(checkpoint)))
    pipe.unet.eval()
    patch, patch_mask, box = crop_pair(image, requested, cfg["resolution"])
    completed = []
    class Cancelled(Exception):
        pass
    def callback(_pipe, _step, _timestep, kwargs):
        if context.cancelled():
            raise Cancelled()
        return kwargs
    try:
        for index in range(count):
            if context.cancelled():
                break
            started = time.perf_counter()
            actual_seed = (seed + index) % 4294967296
            with torch.inference_mode():
                result = pipe(prompt=cfg["prompt"], image=patch, mask_image=patch_mask, height=cfg["resolution"], width=cfg["resolution"],
                              num_inference_steps=30, strength=1.0, generator=torch.Generator(device="cuda").manual_seed(actual_seed),
                              callback_on_step_end=callback).images[0]
            output, changed = compose(image, result, requested, allowed, box)
            sample_id = uid()
            temporary = store.root / "samples" / f".{sample_id}.partial"
            try:
                for name, pixels in (("original", image), ("image", output), ("requested", requested), ("allowed", allowed), ("changed", changed)):
                    save_png(temporary / f"{name}.png", pixels)
                saved = np.array(image8(temporary / "image.png"))
                original = np.array(image8(temporary / "original.png"))
                if not np.array_equal(saved[np.array(allowed) == 0], original[np.array(allowed) == 0]):
                    raise ValueError("저장 후 허용 영역 밖 원본 보존 검사 실패")
                sample = {"id": sample_id, "synthetic": True, "model_id": model["id"], "checkpoint": key, "item_id": model["item_id"],
                          "class_name": model["item"]["class_name"], "source_id": row["id"], "seed": actual_seed,
                          "crop_box": box, "resolution": cfg["resolution"], "created_at": time.time(),
                          "generation_sec": time.perf_counter() - started, "review": {"status": "pending", "mask": "", "reason": ""},
                          "hashes": {p.name: digest(p) for p in temporary.glob("*.png")}}
                write_json(temporary / "manifest.json", sample)
                temporary.rename(store.root / "samples" / sample_id)
                completed.append(sample_id)
                context.emit("datagen_sample", [sample])
                context.emit("progress_updated", [index + 1, count])
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
    except Cancelled:
        pass
    return {"status": "cancelled" if context.cancelled() else "completed", "output": {"samples": completed}}
