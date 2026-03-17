import torch
import torch.nn.functional as F
from tqdm import tqdm

from threestudio.utils.misc import get_device
from threestudio.utils.perceptual import PerceptualLoss
import ui_utils
from threestudio.models.prompt_processors.stable_diffusion_prompt_processor import StableDiffusionPromptProcessor
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.arguments import PipelineParams
from argparse import ArgumentParser
from threestudio.models.guidance.instructpix2pix_guidance import SharedAttentionController


class EditGuidance:
    """
    Guidance wrapper for GaussianEditor edit mode.

    Speed optimisation (pre-generation mode)
    -----------------------------------------
    When `pre_generate=True` (default), ALL edit frames are generated with
    IP2P ONCE before training starts.  During the training loop IP2P is never
    called again — only the cached tensors are used.  This turns a ~9 s/it
    loop into a <0.5 s/it loop.

    When `pre_generate=False` the original behaviour is preserved: IP2P is
    called every `per_editing_step` steps.

    Multi-view consistency strategy
    --------------------------------
    1. Reference view selection: the view whose projected 2D mask has the
       largest area is chosen as the reference.  Its IP2P edit is generated
       first and kept fixed throughout training.
    2. Color-statistics consistency: every other view's rendered output is
       normalised to match the reference edit's per-channel mean/std.
       Lightweight — no extra model needed.
    3. Masked loss: L1 / perceptual loss is computed only inside the projected
       2D mask so that IP2P background changes do not corrupt the rest of the
       scene.
    4. Anchor loss: keeps non-edited Gaussians close to their original state.
    """

    def __init__(self, guidance, gaussian, origin_frames, text_prompt,
                 per_editing_step, edit_begin_step, edit_until_step,
                 lambda_l1, lambda_p,
                 lambda_anchor_color, lambda_anchor_geo,
                 lambda_anchor_scale, lambda_anchor_opacity,
                 train_frames, train_frustums, cams, server,
                 lambda_clip_consistency: float = 0.0,
                 pre_generate: bool = True):
        self.guidance = guidance
        self.gaussian = gaussian
        self.per_editing_step = per_editing_step
        self.edit_begin_step = edit_begin_step
        self.edit_until_step = edit_until_step
        self.lambda_l1 = lambda_l1
        self.lambda_p = lambda_p
        self.lambda_anchor_color = lambda_anchor_color
        self.lambda_anchor_geo = lambda_anchor_geo
        self.lambda_anchor_scale = lambda_anchor_scale
        self.lambda_anchor_opacity = lambda_anchor_opacity
        self.lambda_clip_consistency = lambda_clip_consistency
        self.pre_generate = pre_generate
        self.origin_frames = origin_frames
        self.cams = cams
        self.server = server
        self.train_frames = train_frames
        self.train_frustums = train_frustums
        self.edit_frames = {}
        self.visible = True
        self.prompt_utils = StableDiffusionPromptProcessor(
            {
                "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
                "prompt": text_prompt,
            }
        )()
        self.perceptual_loss = PerceptualLoss().eval().to(get_device())
        self._pipe = PipelineParams(ArgumentParser(description=""))
        self._bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")

        # Reference view: the view with the largest mask area (most informative)
        self._ref_view_index = self._select_reference_view()
        # Cached color statistics of the reference edit (mean, std per channel)
        self._ref_edit_mean = None  # [1, 1, 1, 3] filled after ref edit
        self._ref_edit_std  = None
        print(f"[EditGuidance] Reference view index: {self._ref_view_index}")

        # Pre-generate all edit frames before training starts
        if self.pre_generate:
            self._pre_generate_all_edits()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_mask_2d(self, cam):
        """
        Project the 3D Gaussian mask onto `cam`.
        Returns a [H, W] float tensor in [0,1], or None if no mask is set.
        """
        mask_3d = self.gaussian.mask
        if mask_3d is None or mask_3d.sum() == 0:
            return None
        override = mask_3d.view(-1)[..., None].float().repeat(1, 3)
        semantic = render(cam, self.gaussian, self._pipe, self._bg,
                          override_color=override)["render"]   # [3, H, W]
        mask_2d = torch.norm(semantic, dim=0)                  # [H, W]
        return mask_2d

    def _dilate_mask(self, mask_2d, kernel=5):
        """Binary dilation via max-pool so boundary Gaussians get gradients."""
        m = mask_2d.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W] float32
        # kernel must be odd
        if kernel % 2 == 0:
            kernel += 1
        m = F.max_pool2d(m, kernel_size=kernel, stride=1, padding=kernel // 2)
        return m.squeeze(0).squeeze(0)                 # [H, W]

    @torch.no_grad()
    def _select_reference_view(self):
        """Pick the view whose projected mask covers the most pixels."""
        best_idx, best_area = 0, -1
        for i, cam in enumerate(self.cams):
            m = self._get_mask_2d(cam)
            if m is None:
                continue
            area = (m > 0).sum().item()
            if area > best_area:
                best_area = area
                best_idx = i
        return best_idx

    @torch.no_grad()
    def _run_ip2p(self, rendering, view_index, attn_controller=None, attn_mode="off"):
        """
        Run IP2P on a single view with mask-aware cropping.
        
        Instead of feeding the full image to IP2P (which doesn't know what to edit),
        we crop the region around the target object (expanded by a margin), run IP2P
        on that crop, then composite the edited crop back onto the original image.
        This gives IP2P a much better chance of editing the right thing.
        
        attn_controller / attn_mode: for shared-attention injection.
        
        When shared attention is active, all crops are resized to a fixed
        512x512 before IP2P so that latent spatial dimensions (and thus KV
        sequence lengths) are identical across views.
        """
        cam = self.cams[view_index]
        mask_2d = self._get_mask_2d(cam)  # [H, W] or None
        
        # Fixed IP2P input size when using shared attention (ensures KV dims match)
        FIXED_IP2P_SIZE = 512
        
        if mask_2d is not None and mask_2d.sum() > 0:
            H, W = mask_2d.shape
            binary_mask = (mask_2d > 0)
            
            # Find bounding box of the mask
            ys, xs = torch.where(binary_mask)
            y_min, y_max = ys.min().item(), ys.max().item()
            x_min, x_max = xs.min().item(), xs.max().item()
            
            # Expand bounding box by 50% on each side for context
            box_h = y_max - y_min
            box_w = x_max - x_min
            margin_h = int(box_h * 0.5)
            margin_w = int(box_w * 0.5)
            y_min = max(0, y_min - margin_h)
            y_max = min(H - 1, y_max + margin_h)
            x_min = max(0, x_min - margin_w)
            x_max = min(W - 1, x_max + margin_w)
            
            # Make crop square for better IP2P results
            crop_h = y_max - y_min
            crop_w = x_max - x_min
            crop_size = max(crop_h, crop_w)
            # Center the crop
            cy = (y_min + y_max) // 2
            cx = (x_min + x_max) // 2
            y_min = max(0, cy - crop_size // 2)
            y_max = min(H, y_min + crop_size)
            x_min = max(0, cx - crop_size // 2)
            x_max = min(W, x_min + crop_size)
            # Adjust if we hit boundaries
            if y_max - y_min < crop_size:
                y_min = max(0, y_max - crop_size)
            if x_max - x_min < crop_size:
                x_min = max(0, x_max - crop_size)
            
            # Crop rendering and origin frame: [1, H, W, C] -> [1, crop_h, crop_w, C]
            crop_rendering = rendering[:, y_min:y_max, x_min:x_max, :].clone()
            crop_origin = self.origin_frames[view_index][:, y_min:y_max, x_min:x_max, :].clone()
            
            actual_crop_h, actual_crop_w = crop_rendering.shape[1], crop_rendering.shape[2]
            
            # Resize to fixed size for consistent latent dimensions across views
            if attn_controller is not None and attn_mode != "off":
                # [1, H, W, C] -> [1, C, H, W] for interpolate -> back
                crop_rendering_resized = F.interpolate(
                    crop_rendering.permute(0, 3, 1, 2),
                    (FIXED_IP2P_SIZE, FIXED_IP2P_SIZE),
                    mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)
                crop_origin_resized = F.interpolate(
                    crop_origin.permute(0, 3, 1, 2),
                    (FIXED_IP2P_SIZE, FIXED_IP2P_SIZE),
                    mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)
                
                result = self.guidance(
                    crop_rendering_resized, crop_origin_resized, self.prompt_utils,
                    attn_controller=attn_controller, attn_mode=attn_mode,
                )
                crop_edit_resized = result["edit_images"].detach().clone()
                
                # Resize back to original crop size
                crop_edit = F.interpolate(
                    crop_edit_resized.permute(0, 3, 1, 2),
                    (actual_crop_h, actual_crop_w),
                    mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)
            else:
                result = self.guidance(crop_rendering, crop_origin, self.prompt_utils)
                crop_edit = result["edit_images"].detach().clone()
            
            # Composite: blend edited crop back using dilated mask
            full_edit = rendering.clone()
            mask_crop = self._dilate_mask(mask_2d[y_min:y_max, x_min:x_max] > 0, kernel=11)
            mask_crop_4d = mask_crop.unsqueeze(0).unsqueeze(-1)  # [1, crop_h, crop_w, 1]
            
            # Blend: edited region from IP2P, rest from original rendering
            blended_crop = crop_edit * mask_crop_4d + crop_rendering * (1 - mask_crop_4d)
            full_edit[:, y_min:y_max, x_min:x_max, :] = blended_crop
            
            return full_edit
        else:
            # No mask: fallback to full-image editing
            if attn_controller is not None and attn_mode != "off":
                # Resize to fixed size
                full_h, full_w = rendering.shape[1], rendering.shape[2]
                rendering_resized = F.interpolate(
                    rendering.permute(0, 3, 1, 2),
                    (FIXED_IP2P_SIZE, FIXED_IP2P_SIZE),
                    mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)
                origin_resized = F.interpolate(
                    self.origin_frames[view_index].permute(0, 3, 1, 2),
                    (FIXED_IP2P_SIZE, FIXED_IP2P_SIZE),
                    mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)
                result = self.guidance(
                    rendering_resized, origin_resized, self.prompt_utils,
                    attn_controller=attn_controller, attn_mode=attn_mode,
                )
                edit_resized = result["edit_images"].detach().clone()
                edit = F.interpolate(
                    edit_resized.permute(0, 3, 1, 2),
                    (full_h, full_w),
                    mode="bilinear", align_corners=False,
                ).permute(0, 2, 3, 1)
                return edit
            else:
                result = self.guidance(rendering, self.origin_frames[view_index], self.prompt_utils)
                return result["edit_images"].detach().clone()

    @torch.no_grad()
    def _pre_generate_all_edits(self):
        """
        Generate IP2P edits for ALL views once before training.

        Multi-view consistency via Shared Attention:
        1. Reference view is processed first in "capture" mode — the UNet's
           self-attention K/V at every layer and every denoising step are cached.
        2. All other views are processed in "inject" mode — their self-attention
           K/V are concatenated with the reference's cached K/V, so each view
           "sees" the reference features and produces semantically consistent edits.
        """
        print(f"[EditGuidance] Pre-generating edits for {len(self.cams)} views ...")
        print(f"[EditGuidance] Using Shared Attention injection for multi-view consistency")

        # Create the attention controller (hooks into the UNet)
        attn_ctrl = SharedAttentionController(self.guidance.unet)

        # --- Pass 1: Reference view (capture KV) ---
        attn_ctrl.set_mode("capture")
        ref_cam = self.cams[self._ref_view_index]
        pkg = render(ref_cam, self.gaussian, self._pipe, self._bg)
        ref_rendering = pkg["render"].permute(1, 2, 0).unsqueeze(0)
        ref_edit = self._run_ip2p(
            ref_rendering, self._ref_view_index,
            attn_controller=attn_ctrl, attn_mode="capture",
        )
        self.edit_frames[self._ref_view_index] = ref_edit
        self._ref_edit_mean = ref_edit.mean(dim=(0, 1, 2), keepdim=True)
        self._ref_edit_std  = ref_edit.std(dim=(0, 1, 2), keepdim=True).clamp(min=1e-5)
        print(f"[EditGuidance] Ref edit done (view {self._ref_view_index}), "
              f"KV cached for {len(attn_ctrl.kv_cache)} layers, "
              f"mean={self._ref_edit_mean.squeeze().tolist()}")

        # Update frustum for reference view
        self.train_frustums[self._ref_view_index].remove()
        self.train_frustums[self._ref_view_index] = ui_utils.new_frustums(
            self._ref_view_index, self.train_frames[self._ref_view_index],
            ref_cam, ref_edit, self.visible, self.server,
        )

        # --- Pass 2: All other views (inject reference KV) ---
        attn_ctrl.set_mode("inject")
        other_indices = [i for i in range(len(self.cams)) if i != self._ref_view_index]
        for i in tqdm(other_indices, desc="Pre-generating edits (shared attn)"):
            cam = self.cams[i]
            pkg = render(cam, self.gaussian, self._pipe, self._bg)
            rendering = pkg["render"].permute(1, 2, 0).unsqueeze(0)
            edit = self._run_ip2p(
                rendering, i,
                attn_controller=attn_ctrl, attn_mode="inject",
            )
            self.edit_frames[i] = edit

            # Update frustum display
            self.train_frustums[i].remove()
            self.train_frustums[i] = ui_utils.new_frustums(
                i, self.train_frames[i], cam, edit, self.visible, self.server,
            )

        # Clean up: restore original processors and free KV cache
        attn_ctrl.restore()
        del attn_ctrl

        print(f"[EditGuidance] Pre-generation complete. Training will use cached edits only.")

        # Save all pre-generated edit frames for local inspection
        import os
        from torchvision.utils import save_image
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_views", "edits")
        os.makedirs(save_dir, exist_ok=True)
        for i, edit in self.edit_frames.items():
            img = edit.squeeze(0).permute(2, 0, 1).clamp(0, 1)  # [3, H, W]
            save_image(img, os.path.join(save_dir, f"edit_{i:03d}.png"))
        print(f"[EditGuidance] Saved {len(self.edit_frames)} edit frames to {save_dir}")

    # ------------------------------------------------------------------
    # Main training step
    # ------------------------------------------------------------------

    def __call__(self, rendering, view_index, step):
        self.gaussian.update_learning_rate(step)

        cam = self.cams[view_index]

        if self.pre_generate:
            # ---- Fast path: all edits already cached, no IP2P calls ----
            gt_image = self.edit_frames[view_index]  # [1, H, W, C]
        else:
            # ---- Original path: lazy / periodic IP2P calls ----

            # Ensure reference edit exists first
            if self._ref_edit_mean is None:
                if view_index == self._ref_view_index:
                    ref_edit = self._run_ip2p(rendering, self._ref_view_index)
                else:
                    pkg = render(self.cams[self._ref_view_index], self.gaussian, self._pipe, self._bg)
                    ref_rendering = pkg["render"].permute(1, 2, 0).unsqueeze(0)
                    ref_edit = self._run_ip2p(ref_rendering, self._ref_view_index)
                self.edit_frames[self._ref_view_index] = ref_edit
                self._ref_edit_mean = ref_edit.mean(dim=(0, 1, 2), keepdim=True)
                self._ref_edit_std  = ref_edit.std(dim=(0, 1, 2), keepdim=True).clamp(min=1e-5)
                self.train_frustums[self._ref_view_index].remove()
                self.train_frustums[self._ref_view_index] = ui_utils.new_frustums(
                    self._ref_view_index, self.train_frames[self._ref_view_index],
                    self.cams[self._ref_view_index], ref_edit, self.visible, self.server,
                )

            need_update = (view_index not in self.edit_frames) or (
                self.per_editing_step > 0
                and self.edit_begin_step < step < self.edit_until_step
                and step % self.per_editing_step == 0
                and view_index != self._ref_view_index  # ref view is fixed
            )
            if need_update:
                edit = self._run_ip2p(rendering, view_index)
                self.edit_frames[view_index] = edit
                self.train_frustums[view_index].remove()
                self.train_frustums[view_index] = ui_utils.new_frustums(
                    view_index, self.train_frames[view_index],
                    cam, edit, self.visible, self.server,
                )

            gt_image = self.edit_frames[view_index]  # [1, H, W, C]

        # --- Step 3: build 2D mask and apply masked loss ---
        mask_2d = self._get_mask_2d(cam)
        if mask_2d is not None and mask_2d.sum() > 0:
            mask_float = self._dilate_mask(mask_2d > 0.0, kernel=5)   # [H, W]
            mask_4d = mask_float.unsqueeze(0).unsqueeze(-1)            # [1, H, W, 1]
            rendering_m = rendering * mask_4d
            gt_m        = gt_image  * mask_4d
        else:
            rendering_m = rendering
            gt_m        = gt_image

        loss = (
            self.lambda_l1 * F.l1_loss(rendering_m, gt_m)
            + self.lambda_p * self.perceptual_loss(
                rendering_m.permute(0, 3, 1, 2).contiguous(),
                gt_m.permute(0, 3, 1, 2).contiguous(),
            ).sum()
        )

        # --- Step 4: color-statistics consistency loss (non-reference views only) ---
        if (
            self.lambda_clip_consistency > 0
            and view_index != self._ref_view_index
            and self._ref_edit_mean is not None
            and view_index in self.edit_frames
        ):
            cur_edit = self.edit_frames[view_index]
            cur_mean = cur_edit.mean(dim=(0, 1, 2), keepdim=True)
            cur_std  = cur_edit.std(dim=(0, 1, 2),  keepdim=True).clamp(min=1e-5)
            rendering_norm = (rendering - cur_mean) * (self._ref_edit_std / cur_std) + self._ref_edit_mean
            color_consistency_loss = F.l1_loss(rendering_norm, rendering.detach())
            loss = loss + self.lambda_clip_consistency * color_consistency_loss

        # --- Step 5: anchor loss — protects non-edited Gaussians ---
        if (
            self.lambda_anchor_color > 0
            or self.lambda_anchor_geo > 0
            or self.lambda_anchor_scale > 0
            or self.lambda_anchor_opacity > 0
        ):
            anchor_out = self.gaussian.anchor_loss()
            loss = (
                loss
                + self.lambda_anchor_color   * anchor_out['loss_anchor_color']
                + self.lambda_anchor_geo     * anchor_out['loss_anchor_geo']
                + self.lambda_anchor_opacity * anchor_out['loss_anchor_opacity']
                + self.lambda_anchor_scale   * anchor_out['loss_anchor_scale']
            )

        return loss
