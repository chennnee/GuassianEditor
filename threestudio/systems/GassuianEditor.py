from dataclasses import dataclass, field

from PIL import Image
from tqdm import tqdm
import cv2
import math
import numpy as np
import sys
import shutil
import torch
import threestudio
import os
from threestudio.systems.base import BaseLift3DSystem

from threestudio.utils.typing import *
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import GaussianModel
from gaussiansplatting.scene.cameras import Simple_Camera

from gaussiansplatting.arguments import (
    PipelineParams,
    OptimizationParams
)
from omegaconf import OmegaConf

from argparse import ArgumentParser
from threestudio.utils.misc import get_device
from threestudio.utils.perceptual import PerceptualLoss
from threestudio.utils.sam import LangSAMTextSegmentor, SceneSplatSegmentor


class GaussianEditor(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        gs_source: str = None

        per_editing_step: int = -1
        edit_begin_step: int = 0
        edit_until_step: int = 4000
        
        densify_until_iter: int = 4000
        densify_from_iter: int = 0
        densification_interval: int = 100
        max_densify_percent: float = 0.01

        gs_lr_scaler: float = 1
        gs_final_lr_scaler: float = 1
        color_lr_scaler: float = 1
        opacity_lr_scaler: float = 1
        scaling_lr_scaler: float = 1
        rotation_lr_scaler: float = 1

        # lr
        mask_thres: float = 0.5
        max_grad: float = 1e-7
        min_opacity: float = 0.005

        seg_prompt: str = ""

        # SceneSplat 3D segmentation (bypasses rendering + LangSAM)
        use_scenesplat: bool = False
        scenesplat_checkpoint: str = "/workspace/GaussianEditor/ckpts/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth"

        # cache
        cache_overwrite: bool = True
        cache_dir: str = ""

        # camera quality filter
        alpha_filter_thres: float = 0.3    # views whose mean rendered brightness < this are dropped
        laplacian_thres: float = 2000.0    # views whose Laplacian variance > this are dropped (noise/distortion)

        # target-focused cameras (generated from mask centroid after segmentation)
        n_target_cameras: int = 16         # number of cameras to generate around the target object; 0 = disabled
        target_camera_fovy_deg: float = 49.1

        # anchor
        anchor_weight_init: float = 0.1
        anchor_weight_init_g0: float = 1.0
        anchor_weight_multiplier: float = 2
        training_args: dict = field(default_factory=dict)

    cfg: Config

    def configure(self) -> None:
        self.gaussian = GaussianModel(
            sh_degree=0,
            anchor_weight_init_g0=self.cfg.anchor_weight_init_g0,
            anchor_weight_init=self.cfg.anchor_weight_init,
            anchor_weight_multiplier=self.cfg.anchor_weight_multiplier,
        )
        bg_color = [1, 1, 1] if False else [0, 0, 0]
        self.background_tensor = torch.tensor(
            bg_color, dtype=torch.float32, device="cuda"
        )
        self.edit_frames = {}
        self.origin_frames = {}
        self.perceptual_loss = PerceptualLoss().eval().to(get_device())
        if self.cfg.use_scenesplat and self.cfg.scenesplat_checkpoint:
            self.scenesplat_segmentor = SceneSplatSegmentor(
                checkpoint_path=self.cfg.scenesplat_checkpoint,
                device="cuda:0",
            )
        else:
            self.text_segmentor = LangSAMTextSegmentor().to(get_device())

    @torch.no_grad()
    def update_mask(self, save_name="mask") -> None:
        print(f"Segment with prompt: {self.cfg.seg_prompt}")
        mask_cache_dir = os.path.join(
            self.cache_dir, self.cfg.seg_prompt + f"_{save_name}_{self.view_num}_view"
        )
        gs_mask_path = os.path.join(mask_cache_dir, "gs_mask.pt")
        if not os.path.exists(gs_mask_path) or self.cfg.cache_overwrite:
            if os.path.exists(mask_cache_dir):
                shutil.rmtree(mask_cache_dir)
            os.makedirs(mask_cache_dir)

            if self.cfg.use_scenesplat and hasattr(self, 'scenesplat_segmentor'):
                # SceneSplat: direct 3D segmentation, no rendering needed
                threestudio.info(f"SceneSplat 3D segmentation with prompt: {self.cfg.seg_prompt}")
                selected_mask = self.scenesplat_segmentor.segment(
                    self.gaussian, self.cfg.seg_prompt, threshold=self.cfg.mask_thres
                )
            else:
                # Original LangSAM: render → 2D segment → backproject
                weights = torch.zeros_like(self.gaussian._opacity)
                weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)
                threestudio.info(f"Segmentation with prompt: {self.cfg.seg_prompt}")
                for id in tqdm(self.view_list):
                    cur_path = os.path.join(mask_cache_dir, "{:0>4d}.png".format(id))
                    cur_path_viz = os.path.join(
                        mask_cache_dir, "viz_{:0>4d}.png".format(id)
                    )

                    cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]

                    mask = self.text_segmentor(self.origin_frames[id], self.cfg.seg_prompt)[
                        0
                    ].to(get_device())

                    mask_to_save = (
                            mask[0]
                            .cpu()
                            .detach()[..., None]
                            .repeat(1, 1, 3)
                            .numpy()
                            .clip(0.0, 1.0)
                            * 255.0
                    ).astype(np.uint8)
                    cv2.imwrite(cur_path, mask_to_save)

                    masked_image = self.origin_frames[id].detach().clone()[0]
                    masked_image[mask[0].bool()] *= 0.3
                    masked_image_to_save = (
                            masked_image.cpu().detach().numpy().clip(0.0, 1.0) * 255.0
                    ).astype(np.uint8)
                    masked_image_to_save = cv2.cvtColor(
                        masked_image_to_save, cv2.COLOR_RGB2BGR
                    )
                    cv2.imwrite(cur_path_viz, masked_image_to_save)
                    self.gaussian.apply_weights(cur_cam, weights, weights_cnt, mask)

                weights /= weights_cnt + 1e-7

                selected_mask = weights > self.cfg.mask_thres
                selected_mask = selected_mask[:, 0]

            torch.save(selected_mask, gs_mask_path)
        else:
            print("load cache")
            for id in tqdm(self.view_list):
                cur_path = os.path.join(mask_cache_dir, "{:0>4d}.png".format(id))
                cur_mask = cv2.imread(cur_path)
                cur_mask = torch.tensor(
                    cur_mask / 255, device="cuda", dtype=torch.float32
                )[..., 0][None]
            selected_mask = torch.load(gs_mask_path)

        self.gaussian.set_mask(selected_mask)
        self.gaussian.apply_grad_mask(selected_mask)
        self.generate_and_add_target_cameras()

    def on_fit_start(self) -> None:
        self.prefilter_cameras()

    def on_validation_epoch_end(self):
        pass

    def forward(self, batch: Dict[str, Any], renderbackground=None, local=False) -> Dict[str, Any]:
        if renderbackground is None:
            renderbackground = self.background_tensor
        images = []
        depths = []
        semantics = []
        masks = []
        self.viewspace_point_list = []
        self.gaussian.localize = local

        for id, cam in enumerate(batch["camera"]):
            render_pkg = render(cam, self.gaussian, self.pipe, renderbackground)
            image, viewspace_point_tensor, _, radii = (
                render_pkg["render"],
                render_pkg["viewspace_points"],
                render_pkg["visibility_filter"],
                render_pkg["radii"],
            )
            self.viewspace_point_list.append(viewspace_point_tensor)

            if id == 0:
                self.radii = radii
            else:
                self.radii = torch.max(radii, self.radii)

            depth = render_pkg["depth_3dgs"]
            depth = depth.permute(1, 2, 0)

            semantic_map = render(
                cam,
                self.gaussian,
                self.pipe,
                renderbackground,
                override_color=self.gaussian.mask[..., None].float().repeat(1, 3),
            )["render"]
            semantic_map = torch.norm(semantic_map, dim=0)
            semantic_map = semantic_map > 0.8
            semantic_map_viz = image.detach().clone()
            semantic_map_viz = semantic_map_viz.permute(
                1, 2, 0
            )  # 3 512 512 to 512 512 3
            semantic_map_viz[semantic_map] = 0.40 * semantic_map_viz[
                semantic_map
            ] + 0.60 * torch.tensor([1.0, 0.0, 0.0], device="cuda")
            semantic_map_viz = semantic_map_viz.permute(
                2, 0, 1
            )  # 512 512 3 to 3 512 512

            semantics.append(semantic_map_viz)
            masks.append(semantic_map)
            image = image.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)

        self.gaussian.localize = False  # reverse

        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        semantics = torch.stack(semantics, dim=0)
        masks = torch.stack(masks, dim=0)

        render_pkg["semantic"] = semantics
        render_pkg["masks"] = masks
        self.visibility_filter = self.radii > 0.0
        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)
        return {
            **render_pkg,
        }

    def render_all_view(self, cache_name):
        cache_dir = os.path.join(self.cache_dir, cache_name)
        os.makedirs(cache_dir, exist_ok=True)
        with torch.no_grad():
            for id in tqdm(self.view_list):
                cur_path = os.path.join(cache_dir, "{:0>4d}.png".format(id))
                if not os.path.exists(cur_path) or self.cfg.cache_overwrite:
                    cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
                    cur_batch = {
                        "index": id,
                        "camera": [cur_cam],
                        "height": self.trainer.datamodule.train_dataset.height,
                        "width": self.trainer.datamodule.train_dataset.width,
                    }
                    out = self(cur_batch)["comp_rgb"]
                    out_to_save = (
                            out[0].cpu().detach().numpy().clip(0.0, 1.0) * 255.0
                    ).astype(np.uint8)
                    out_to_save = cv2.cvtColor(out_to_save, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(cur_path, out_to_save)
                cached_image = cv2.cvtColor(cv2.imread(cur_path), cv2.COLOR_BGR2RGB)
                self.origin_frames[id] = torch.tensor(
                    cached_image / 255, device="cuda", dtype=torch.float32
                )[None]

    @torch.no_grad()
    def prefilter_cameras(self):
        """Drop low-quality views before editing training.

        Two indicators are checked per view:
          1. Mean brightness  — black-background renders with low mean brightness
             indicate the camera is looking at mostly empty / untrained space.
          2. Laplacian variance — very high high-frequency energy indicates
             geometric distortion or floating-Gaussian noise (over-exposed views).

        Both thresholds can be disabled by setting them to 0 / a very large value.
        """
        if self.cfg.alpha_filter_thres <= 0 and self.cfg.laplacian_thres <= 0:
            return
        valid_ids = []
        rejected_brightness = 0
        rejected_laplacian = 0
        for id in tqdm(self.view_list, desc="Prefiltering cameras"):
            cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]
            render_pkg = render(cur_cam, self.gaussian, self.pipe, self.background_tensor)
            rendered = render_pkg["render"]  # [3, H, W], float32 in [0,1]

            # --- indicator 1: mean brightness (proxy for alpha coverage) ---
            # Note: diff_gaussian_rasterization does not expose a separate alpha
            # channel. With a black background (bg=[0,0,0]), empty pixels render
            # as (0,0,0), so mean RGB brightness is a reliable proxy for scene
            # coverage — low brightness ≈ camera looking at mostly empty space.
            if self.cfg.alpha_filter_thres > 0:
                mean_brightness = rendered.mean().item()
                if mean_brightness < self.cfg.alpha_filter_thres:
                    rejected_brightness += 1
                    continue

            # --- indicator 2: Laplacian variance (high-frequency noise) ---
            if self.cfg.laplacian_thres > 0:
                img_np = (rendered.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if lap_var > self.cfg.laplacian_thres:
                    rejected_laplacian += 1
                    continue

            valid_ids.append(id)

        n_before, n_after = len(self.view_list), len(valid_ids)
        threestudio.info(
            f"Camera prefilter: {n_after}/{n_before} views passed "
            f"(rejected by brightness={rejected_brightness}, laplacian={rejected_laplacian})"
        )
        if n_after == 0:
            threestudio.warn("All views filtered out — keeping original list.")
            return
        if n_after < 10:
            threestudio.warn(
                f"Only {n_after} views passed the filter (< 10). "
                "Keeping original list to avoid under-constrained optimization."
            )
            return

        self.view_list = valid_ids
        self.view_num = len(valid_ids)
        # keep the dataset's index list in sync so collate() only samples good views
        self.trainer.datamodule.train_dataset.n2n_view_index = valid_ids
        self.trainer.datamodule.train_dataset.view_index_stack = valid_ids.copy()

    @torch.no_grad()
    def generate_and_add_target_cameras(self):
        """Generate cameras focused on the segmented target object and add them to the view list.

        After SceneSplat segmentation, we know the 3D positions of target Gaussians.
        We compute their centroid and bounding radius, then generate n_target_cameras
        cameras uniformly distributed on a sphere around the target, all looking at it.
        These cameras are appended to the scene camera list and view_list so that
        training steps sample them alongside (or instead of) the original cameras.
        """
        if self.cfg.n_target_cameras <= 0:
            return
        if not hasattr(self.gaussian, 'mask') or self.gaussian.mask is None:
            threestudio.warn("generate_and_add_target_cameras: no mask set, skipping.")
            return

        mask = self.gaussian.mask
        if mask.sum() == 0:
            threestudio.warn("generate_and_add_target_cameras: mask is empty, skipping.")
            return

        # --- compute target centroid and camera radius ---
        target_xyz = self.gaussian._xyz[mask].detach()           # [M, 3]
        target_center = target_xyz.mean(dim=0).cpu().numpy()     # (3,)
        # use 3× the object's bounding radius so the whole object fits in frame
        obj_radius = (target_xyz - target_xyz.mean(dim=0)).norm(dim=1).max().item()
        cam_radius = max(obj_radius * 3.0, self.cameras_extent * 0.3)

        h = self.trainer.datamodule.train_dataset.height
        w = self.trainer.datamodule.train_dataset.width
        fovy = math.radians(self.cfg.target_camera_fovy_deg)
        fovx = 2 * math.atan(math.tan(fovy / 2) * (w / h))

        # --- Fibonacci sphere sampling around target ---
        golden_ratio = (1 + math.sqrt(5)) / 2
        new_cameras = []
        for i in range(self.cfg.n_target_cameras):
            theta = 2 * math.pi * i / golden_ratio
            phi = math.acos(1 - 2 * (i + 0.5) / self.cfg.n_target_cameras)

            cam_pos = target_center + cam_radius * np.array([
                math.sin(phi) * math.cos(theta),
                -math.cos(phi),                   # y-down convention
                math.sin(phi) * math.sin(theta),
            ])

            z_axis = target_center - cam_pos
            z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

            world_up = np.array([0.0, -1.0, 0.0])
            if abs(np.dot(z_axis, world_up)) > 0.99:
                world_up = np.array([1.0, 0.0, 0.0])

            x_axis = np.cross(world_up, z_axis)
            x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

            R = np.stack([x_axis, y_axis, z_axis], axis=0)
            T = -R @ cam_pos

            cam = Simple_Camera(0, R, T, fovx, fovy, h, w, "target", 0)
            new_cameras.append(cam)

        # --- append to scene camera list and view_list ---
        scene_cameras = self.trainer.datamodule.train_dataset.scene.cameras
        base_idx = len(scene_cameras)
        scene_cameras.extend(new_cameras)

        new_ids = list(range(base_idx, base_idx + len(new_cameras)))
        self.view_list = self.view_list + new_ids
        self.view_num = len(self.view_list)
        self.trainer.datamodule.train_dataset.n2n_view_index = self.view_list
        self.trainer.datamodule.train_dataset.view_index_stack = self.view_list.copy()

        # render and cache origin_frames for the new cameras so training_step
        # can access self.origin_frames[cur_index] without KeyError
        for cam_id, cam in zip(new_ids, new_cameras):
            render_pkg = render(cam, self.gaussian, self.pipe, self.background_tensor)
            rendered = render_pkg["render"].permute(1, 2, 0).clamp(0, 1)  # [H,W,3]
            self.origin_frames[cam_id] = rendered.unsqueeze(0)  # [1,H,W,3]

        threestudio.info(
            f"Added {len(new_cameras)} target-focused cameras "
            f"(center={target_center.round(3)}, radius={cam_radius:.3f}). "
            f"Total views: {self.view_num}"
        )

    def on_before_optimizer_step(self, optimizer):
        with torch.no_grad():
            if self.true_global_step < self.cfg.densify_until_iter:
                viewspace_point_tensor_grad = torch.zeros_like(
                    self.viewspace_point_list[0]
                )
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = (
                            viewspace_point_tensor_grad
                            + self.viewspace_point_list[idx].grad
                    )
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                    self.gaussian.max_radii2D[self.visibility_filter],
                    self.radii[self.visibility_filter],
                )
                self.gaussian.add_densification_stats(
                    viewspace_point_tensor_grad, self.visibility_filter
                )
                # Densification
                if (
                        self.true_global_step >= self.cfg.densify_from_iter
                        and self.true_global_step % self.cfg.densification_interval == 0
                ):  # 500 100
                    self.gaussian.densify_and_prune(
                        self.cfg.max_grad,
                        self.cfg.max_densify_percent,
                        self.cfg.min_opacity,
                        self.cameras_extent,
                        5,
                    )

    def validation_step(self, batch, batch_idx):
        batch["camera"] = [
            self.trainer.datamodule.train_dataset.scene.cameras[idx]
            for idx in batch["index"]
        ]
        out = self(batch)
        for idx in range(len(batch["index"])):
            cam_index = batch["index"][idx].item()
            self.save_image_grid(
                f"it{self.true_global_step}-{batch['index'][idx]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": self.origin_frames[cam_index][0],
                            "kwargs": {"data_format": "HWC"},
                        },
                        {
                            "type": "rgb",
                            "img": self.edit_frames[cam_index][0]
                            if cam_index in self.edit_frames
                            else torch.zeros_like(self.origin_frames[cam_index][0]),
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                ),
                name=f"validation_step_{idx}",
                step=self.true_global_step,
            )
            self.save_image_grid(
                f"render_it{self.true_global_step}-{batch['index'][idx]}.png",
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][idx],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][idx],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["semantic"][idx].moveaxis(0, -1),
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "semantic" in out
                    else []
                ),
                name=f"validation_step_render_{idx}",
                step=self.true_global_step,
            )

    def test_step(self, batch, batch_idx):
        only_rgb = True  # TODO add depth test step
        bg_color = [1, 1, 1] if False else [0, 0, 0]
        batch["camera"] = [
            self.trainer.datamodule.val_dataset.scene.cameras[batch["index"]]
        ]
        testbackground_tensor = torch.tensor(
            bg_color, dtype=torch.float32, device="cuda"
        )

        out = self(batch, testbackground_tensor)
        if only_rgb:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                ),
                name="test_step",
                step=self.true_global_step,
            )
        else:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out["depth"][0],
                            "kwargs": {},
                        }
                    ]
                    if "depth" in out
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="test_step",
                step=self.true_global_step,
            )

    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=2,
            name="test",
            step=self.true_global_step,
        )
        save_list = []
        for index, image in sorted(self.edit_frames.items(), key=lambda item: item[0]):
            save_list.append(
                {
                    "type": "rgb",
                    "img": image[0],
                    "kwargs": {"data_format": "HWC"},
                },
            )
        if len(save_list) > 0:
            self.save_image_grid(
                f"edited_images.png",
                save_list,
                name="edited_images",
                step=self.true_global_step,
            )
        save_list = []
        for index, image in sorted(
                self.origin_frames.items(), key=lambda item: item[0]
        ):
            save_list.append(
                {
                    "type": "rgb",
                    "img": image[0],
                    "kwargs": {"data_format": "HWC"},
                },
            )
        self.save_image_grid(
            f"origin_images.png",
            save_list,
            name="origin",
            step=self.true_global_step,
        )

        save_path = self.get_save_path(f"last.ply")
        self.gaussian.save_ply(save_path)

    def configure_optimizers(self):
        self.parser = ArgumentParser(description="Training script parameters")
        self.view_list = self.trainer.datamodule.train_dataset.n2n_view_index
        self.view_num = len(self.view_list)
        opt = OptimizationParams(self.parser, self.trainer.max_steps, self.cfg.gs_lr_scaler, self.cfg.gs_final_lr_scaler, self.cfg.color_lr_scaler,
                                 self.cfg.opacity_lr_scaler, self.cfg.scaling_lr_scaler, self.cfg.rotation_lr_scaler, )
        self.gaussian.load_ply(self.cfg.gs_source)
        self.gaussian.max_radii2D = torch.zeros(
            (self.gaussian.get_xyz.shape[0]), device="cuda"
        )
        self.cameras_extent = self.trainer.datamodule.train_dataset.scene.cameras_extent
        self.gaussian.spatial_lr_scale = self.cameras_extent

        self.pipe = PipelineParams(self.parser)
        opt = OmegaConf.create(vars(opt))
        opt.update(self.cfg.training_args)
        self.gaussian.training_setup(opt)

        ret = {
            "optimizer": self.gaussian.optimizer,
        }

        return ret
