"""Real tiny Diffusers UNet: 9-channel LoRA backward and safe adapter reload.

This is a CPU code contract, not pretrained GPU or generation-quality evidence.
"""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gui"))


@unittest.skipUnless(importlib.util.find_spec("diffusers") and importlib.util.find_spec("peft"), "Data Gen optional runtime not installed")
class DiffusersContractTests(unittest.TestCase):
    def test_inpainting_lora_backward_and_exact_reload(self):
        import torch
        from diffusers import UNet2DConditionModel, DDPMScheduler
        import peft
        from safetensors.torch import save_file, load_file
        from core.datagen_learning import add_adapter, area_loss
        from types import SimpleNamespace
        torch.set_num_threads(1)
        torch.manual_seed(7)
        model = UNet2DConditionModel(sample_size=8, in_channels=9, out_channels=4,
            layers_per_block=1, block_out_channels=(16, 32), norm_num_groups=8,
            down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"), cross_attention_dim=16, attention_head_dim=4)
        base = {k: v.clone() for k, v in model.state_dict().items()}
        model.requires_grad_(False)
        add_adapter(SimpleNamespace(unet=model), {"rank": 2}, peft)
        optimizer = torch.optim.AdamW([v for v in model.parameters() if v.requires_grad], lr=.01)
        latent = torch.randn(1, 4, 8, 8)
        mask = torch.zeros(1, 1, 8, 8)
        mask[:, :, 2:4, 3:5] = 1
        noise = torch.randn_like(latent)
        embeddings = torch.randn(1, 3, 16)
        timestep = torch.tensor([500])
        schedule = DDPMScheduler(prediction_type="v_prediction")
        inputs = torch.cat([schedule.add_noise(latent, noise, timestep), mask, latent * (1-mask)], dim=1)
        predicted = model(inputs, timestep, encoder_hidden_states=embeddings).sample
        before = {k: v.clone() for k, v in peft.get_peft_model_state_dict(model).items()}
        loss, defect, background = area_loss(predicted, schedule.get_velocity(latent, noise, timestep), mask, .1)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        optimizer.step()
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in peft.get_peft_model_state_dict(model).items()))
        model.eval()
        expected = model(inputs, timestep, encoder_hidden_states=embeddings).sample.detach()
        restored = UNet2DConditionModel.from_config(model.config)
        restored.load_state_dict(base)
        restored.requires_grad_(False)
        add_adapter(SimpleNamespace(unet=restored), {"rank": 2}, peft)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "adapter.safetensors")
            save_file({k: v.detach().contiguous() for k, v in peft.get_peft_model_state_dict(model).items()}, path)
            peft.set_peft_model_state_dict(restored, load_file(path))
        restored.eval()
        actual = restored(inputs, timestep, encoder_hidden_states=embeddings).sample.detach()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for name, parameter in model.named_parameters():
            if "lora_" not in name:
                original = name.replace(".base_layer", "")
                torch.testing.assert_close(parameter, base[original], rtol=0, atol=0)
        self.assertGreater(float(defect), 0)
        self.assertGreater(float(background), 0)

    def test_area_normalization_does_not_dilute_small_defect(self):
        import torch
        from core.datagen_learning import area_loss
        mask = torch.zeros(1, 1, 32, 32)
        mask[:, :, 3, 4] = 1
        target = torch.zeros(1, 4, 32, 32)
        prediction = mask.expand_as(target) * 2
        total, defect, background = area_loss(prediction, target, mask, .1)
        self.assertEqual(float(total), 4)
        self.assertEqual(float(defect), 4)
        self.assertEqual(float(background), 0)


if __name__ == "__main__":
    unittest.main()
