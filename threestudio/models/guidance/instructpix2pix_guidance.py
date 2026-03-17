import os
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/.cache/huggingface/hub"

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionInstructPix2PixPipeline
from diffusers.utils.import_utils import is_xformers_available
from tqdm import tqdm

import threestudio
from threestudio.models.prompt_processors.base import PromptProcessorOutput
from threestudio.utils.base import BaseObject
from threestudio.utils.misc import C, parse_version
from threestudio.utils.typing import *


class SharedAttentionController:
    """
    Controls reference-attention injection for multi-view consistent editing.

    Usage:
        controller = SharedAttentionController(unet)
        # Step 1: run reference view to capture KV
        controller.set_mode("capture")
        run_denoising(...)
        # Step 2: run other views with injected KV
        controller.set_mode("inject")
        run_denoising(...)
        # Step 3: restore original processors
        controller.restore()

    During "inject" mode, each self-attention layer concatenates the
    reference K/V with the current view's K/V so that:
        attn_output = softmax(Q_cur @ [K_cur, K_ref]^T) @ [V_cur, V_ref]
    This lets the current view "see" the reference features while keeping
    its own spatial structure (Q is unchanged).
    """

    def __init__(self, unet):
        self.unet = unet
        # Save original processors so we can restore later
        self._original_processors = dict(unet.attn_processors)
        # Identify self-attention layer names (those without "encoder" in name,
        # i.e. not cross-attention)
        self._self_attn_names = [
            name for name in unet.attn_processors.keys()
            if "attn1" in name  # attn1 = self-attention, attn2 = cross-attention
        ]
        # KV cache: {layer_name: {step_index: (K, V)}}
        self.kv_cache = {}
        self._step_counter = 0
        self._mode = "off"  # "off", "capture", "inject"

    def set_mode(self, mode: str):
        """Switch between 'capture', 'inject', or 'off'."""
        assert mode in ("off", "capture", "inject")
        self._mode = mode
        self._step_counter = 0
        if mode == "capture":
            self.kv_cache = {}
        self._install_processors()

    def step_done(self):
        """Call after each denoising step to advance the step counter."""
        self._step_counter += 1

    def restore(self):
        """Restore original attention processors."""
        self._mode = "off"
        self.unet.set_attn_processor(self._original_processors)

    def _install_processors(self):
        """Install custom processors for self-attention layers."""
        if self._mode == "off":
            self.unet.set_attn_processor(self._original_processors)
            return

        new_processors = {}
        for name, proc in self._original_processors.items():
            if name in self._self_attn_names:
                new_processors[name] = _RefAttnProcessor(
                    controller=self,
                    layer_name=name,
                    original_processor=proc,
                )
            else:
                new_processors[name] = proc
        self.unet.set_attn_processor(new_processors)


class _RefAttnProcessor:
    """
    Custom attention processor that captures or injects reference KV.
    Only used for self-attention (attn1) layers.
    """

    def __init__(self, controller: SharedAttentionController, layer_name: str,
                 original_processor):
        self.controller = controller
        self.layer_name = layer_name
        self.original_processor = original_processor

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args, **kwargs,
    ):
        # This should only be called for self-attention (encoder_hidden_states=None)
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        step = self.controller._step_counter
        mode = self.controller._mode

        if mode == "capture":
            # Store KV for this layer at this step
            if self.layer_name not in self.controller.kv_cache:
                self.controller.kv_cache[self.layer_name] = {}
            self.controller.kv_cache[self.layer_name][step] = (
                key.detach().clone(),
                value.detach().clone(),
            )

        elif mode == "inject":
            # Concatenate reference KV with current KV
            cache = self.controller.kv_cache.get(self.layer_name, {})
            if step in cache:
                ref_key, ref_value = cache[step]
                # key/value shape: [batch*heads, seq_len, dim] after head_to_batch_dim
                # But we haven't called head_to_batch_dim yet, so shape is [batch, seq, dim]
                # Concatenate along sequence dimension
                key = torch.cat([key, ref_key], dim=1)
                value = torch.cat([value, ref_value], dim=1)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj + dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


@threestudio.register("stable-diffusion-instructpix2pix-guidance")
class InstructPix2PixGuidance(BaseObject):
    @dataclass
    class Config(BaseObject.Config):
        cache_dir: Optional[str] = None
        ddim_scheduler_name_or_path: str = "/workspace/.cache/huggingface/hub/models--CompVis--stable-diffusion-v1-4/snapshots/133a221b8aa7292a167afc5127cb63fb5005638b"
        ip2p_name_or_path: str = "timbrooks/instruct-pix2pix"

        enable_memory_efficient_attention: bool = False
        enable_sequential_cpu_offload: bool = False
        enable_attention_slicing: bool = False
        enable_channels_last_format: bool = False
        guidance_scale: float = 7.5
        condition_scale: float = 1.5
        grad_clip: Optional[
            Any
        ] = None  # field(default_factory=lambda: [0, 2.0, 8.0, 1000])
        half_precision_weights: bool = True

        fixed_size: int = -1

        min_step_percent: float = 0.02
        max_step_percent: float = 0.98

        diffusion_steps: int = 20

        use_sds: bool = False

    cfg: Config

    def configure(self) -> None:
        threestudio.info(f"Loading InstructPix2Pix ...")

        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )

        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": self.weights_dtype,
            "cache_dir": self.cfg.cache_dir,
        }

        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            self.cfg.ip2p_name_or_path, **pipe_kwargs, local_files_only=True
        ).to(self.device)
        self.scheduler = DDIMScheduler.from_pretrained(
            self.cfg.ddim_scheduler_name_or_path,
            subfolder="scheduler",
            local_files_only=True,
            torch_dtype=self.weights_dtype,
            cache_dir=self.cfg.cache_dir,
        )
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)

        if self.cfg.enable_memory_efficient_attention:
            if parse_version(torch.__version__) >= parse_version("2"):
                threestudio.info(
                    "PyTorch2.0 uses memory efficient attention by default."
                )
            elif not is_xformers_available():
                threestudio.warn(
                    "xformers is not available, memory efficient attention is not enabled."
                )
            else:
                self.pipe.enable_xformers_memory_efficient_attention()

        if self.cfg.enable_sequential_cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        if self.cfg.enable_attention_slicing:
            self.pipe.enable_attention_slicing(1)

        if self.cfg.enable_channels_last_format:
            self.pipe.unet.to(memory_format=torch.channels_last)

        # Create model
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.eval()

        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value

        self.alphas: Float[Tensor, "..."] = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = None

        threestudio.info(f"Loaded InstructPix2Pix!")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        latents: Float[Tensor, "..."],
        t: Float[Tensor, "..."],
        encoder_hidden_states: Float[Tensor, "..."],
    ) -> Float[Tensor, "..."]:
        input_dtype = latents.dtype
        return self.unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
        ).sample.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def encode_cond_images(
        self, imgs: Float[Tensor, "B 3 H W"]
    ) -> Float[Tensor, "B 4 DH DW"]:
        input_dtype = imgs.dtype
        imgs = imgs * 2.0 - 1.0
        posterior = self.vae.encode(imgs.to(self.weights_dtype)).latent_dist
        latents = posterior.mode()
        uncond_image_latents = torch.zeros_like(latents)
        latents = torch.cat([latents, latents, uncond_image_latents], dim=0)
        return latents.to(input_dtype)

    @torch.cuda.amp.autocast(enabled=False)
    def decode_latents(
        self, latents: Float[Tensor, "B 4 DH DW"]
    ) -> Float[Tensor, "B 3 H W"]:
        input_dtype = latents.dtype
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents.to(self.weights_dtype)).sample
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image.to(input_dtype)

    def edit_latents(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond_latents: Float[Tensor, "B 4 DH DW"],
        t: Int[Tensor, "B"],
    ) -> Float[Tensor, "B 4 DH DW"]:
        self.scheduler.config.num_train_timesteps = t.item()
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, t)  # type: ignore
            threestudio.debug("Start editing...")
            # sections of code used from https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py
            for i, t in enumerate(self.scheduler.timesteps):
                # predict the noise residual with unet, NO grad!
                with torch.no_grad():
                    # pred noise
                    latent_model_input = torch.cat([latents] * 3)
                    latent_model_input = torch.cat(
                        [latent_model_input, image_cond_latents], dim=1
                    )

                    noise_pred = self.forward_unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    )

                # perform classifier-free guidance
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(
                    3
                )
                noise_pred = (
                    noise_pred_uncond
                    + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
                    + self.cfg.condition_scale * (noise_pred_image - noise_pred_uncond)
                )

                # get previous sample, continue loop
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            threestudio.debug("Editing finished.")
        return latents

    def edit_latents_ref_attn(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond_latents: Float[Tensor, "B 4 DH DW"],
        t: Int[Tensor, "B"],
        attn_controller: SharedAttentionController = None,
        attn_mode: str = "off",
    ) -> Float[Tensor, "B 4 DH DW"]:
        """
        Same as edit_latents but with shared-attention control.
        attn_mode: "capture" to record reference KV, "inject" to use them.
        
        NOTE: The caller is responsible for calling attn_controller.set_mode()
        before and attn_controller.restore() after the full batch of views.
        This method only resets the step counter per call.
        """
        self.scheduler.config.num_train_timesteps = t.item()
        self.scheduler.set_timesteps(self.cfg.diffusion_steps)

        # Reset step counter for this denoising run (KV cache is indexed by step)
        if attn_controller is not None:
            attn_controller._step_counter = 0

        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents = self.scheduler.add_noise(latents, noise, t)
            threestudio.debug("Start editing (ref_attn mode=%s)...", attn_mode)

            for i, t_step in enumerate(self.scheduler.timesteps):
                with torch.no_grad():
                    latent_model_input = torch.cat([latents] * 3)
                    latent_model_input = torch.cat(
                        [latent_model_input, image_cond_latents], dim=1
                    )
                    noise_pred = self.forward_unet(
                        latent_model_input, t_step,
                        encoder_hidden_states=text_embeddings,
                    )

                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred = (
                    noise_pred_uncond
                    + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
                    + self.cfg.condition_scale * (noise_pred_image - noise_pred_uncond)
                )
                latents = self.scheduler.step(noise_pred, t_step, latents).prev_sample

                if attn_controller is not None:
                    attn_controller.step_done()

            threestudio.debug("Editing finished (ref_attn).")

        return latents

    def compute_grad_sds(
        self,
        text_embeddings: Float[Tensor, "BB 77 768"],
        latents: Float[Tensor, "B 4 DH DW"],
        image_cond_latents: Float[Tensor, "B 4 DH DW"],
        t: Int[Tensor, "B"],
    ):
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)  # TODO: use torch generator
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 3)
            latent_model_input = torch.cat(
                [latent_model_input, image_cond_latents], dim=1
            )

            noise_pred = self.forward_unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )

        noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
        noise_pred = (
            noise_pred_uncond
            + self.cfg.guidance_scale * (noise_pred_text - noise_pred_image)
            + self.cfg.condition_scale * (noise_pred_image - noise_pred_uncond)
        )

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        grad = w * (noise_pred - noise)
        return grad

    def __call__(
        self,
        rgb: Float[Tensor, "B H W C"],
        cond_rgb: Float[Tensor, "B H W C"],
        prompt_utils: PromptProcessorOutput,
        attn_controller: SharedAttentionController = None,
        attn_mode: str = "off",
        **kwargs,
    ):
        batch_size, H, W, _ = rgb.shape

        rgb_BCHW = rgb.permute(0, 3, 1, 2)
        latents: Float[Tensor, "B 4 DH DW"]
        if self.cfg.fixed_size > 0:
            RH, RW = self.cfg.fixed_size, self.cfg.fixed_size
        else:
            RH, RW = H // 8 * 8, W // 8 * 8
        rgb_BCHW_HW8 = F.interpolate(
            rgb_BCHW, (RH, RW), mode="bilinear", align_corners=False
        )
        latents = self.encode_images(rgb_BCHW_HW8)

        cond_rgb_BCHW = cond_rgb.permute(0, 3, 1, 2)
        cond_rgb_BCHW_HW8 = F.interpolate(
            cond_rgb_BCHW,
            (RH, RW),
            mode="bilinear",
            align_corners=False,
        )
        cond_latents = self.encode_cond_images(cond_rgb_BCHW_HW8)

        temp = torch.zeros(1).to(rgb.device)
        text_embeddings = prompt_utils.get_text_embeddings(temp, temp, temp, False)
        text_embeddings = torch.cat(
            [text_embeddings, text_embeddings[-1:]], dim=0
        )  # [positive, negative, negative]

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(
            self.min_step,
            self.max_step + 1,
            [batch_size],
            dtype=torch.long,
            device=self.device,
        )

        if self.cfg.use_sds:
            grad = self.compute_grad_sds(text_embeddings, latents, cond_latents, t)
            grad = torch.nan_to_num(grad)
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)
            target = (latents - grad).detach()
            loss_sds = 0.5 * F.mse_loss(latents, target, reduction="sum") / batch_size
            return {
                "loss_sds": loss_sds,
                "grad_norm": grad.norm(),
                "min_step": self.min_step,
                "max_step": self.max_step,
            }
        else:
            if attn_controller is not None and attn_mode != "off":
                edit_latents = self.edit_latents_ref_attn(
                    text_embeddings, latents, cond_latents, t,
                    attn_controller=attn_controller, attn_mode=attn_mode,
                )
            else:
                edit_latents = self.edit_latents(text_embeddings, latents, cond_latents, t)
            edit_images = self.decode_latents(edit_latents)
            edit_images = F.interpolate(edit_images, (H, W), mode="bilinear")

            return {"edit_images": edit_images.permute(0, 2, 3, 1)}

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):
        # clip grad for stable training as demonstrated in
        # Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation
        # http://arxiv.org/abs/2303.15413
        if self.cfg.grad_clip is not None:
            self.grad_clip_val = C(self.cfg.grad_clip, epoch, global_step)

        self.set_min_max_steps(
            min_step_percent=C(self.cfg.min_step_percent, epoch, global_step),
            max_step_percent=C(self.cfg.max_step_percent, epoch, global_step),
        )


if __name__ == "__main__":
    from threestudio.utils.config import ExperimentConfig, load_config
    from threestudio.utils.typing import Optional

    cfg = load_config("configs/debugging/instructpix2pix.yaml")
    guidance = threestudio.find(cfg.system.guidance_type)(cfg.system.guidance)
    prompt_processor = threestudio.find(cfg.system.prompt_processor_type)(
        cfg.system.prompt_processor
    )
    rgb_image = cv2.imread("assets/face.jpg")[:, :, ::-1].copy() / 255
    rgb_image = torch.FloatTensor(rgb_image).unsqueeze(0).to(guidance.device)
    prompt_utils = prompt_processor()
    guidance_out = guidance(rgb_image, rgb_image, prompt_utils)
    edit_image = (
        (
            guidance_out["edit_images"][0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .clip(0, 1)
            .numpy()
            * 255
        )
        .astype(np.uint8)[:, :, ::-1]
        .copy()
    )
    import os

    os.makedirs(".threestudio_cache", exist_ok=True)
    cv2.imwrite(".threestudio_cache/edit_image.jpg", edit_image)
