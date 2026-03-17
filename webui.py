import os
import threading
os.environ["HUGGINGFACE_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/workspace/.cache/huggingface/hub"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/.cache/huggingface/hub"
import time
import numpy as np
import torch
import torchvision
import rembg
from gaussiansplatting.scene.colmap_loader import qvec2rotmat
from gaussiansplatting.scene.cameras import Simple_Camera
from threestudio.utils.dpt import DPT
from torchvision.ops import masks_to_boxes
from gaussiansplatting.utils.graphics_utils import fov2focal

import viser
import viser.transforms as tf
from dataclasses import dataclass, field
from viser.theme import TitlebarButton, TitlebarConfig, TitlebarImage

from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import sys
import shutil
import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from kornia.geometry.quaternion import Quaternion

from threestudio.utils.typing import *
from threestudio.utils.transform import rotate_gaussians
from gaussiansplatting.gaussian_renderer import render, point_cloud_render
from gaussiansplatting.scene import GaussianModel

from gaussiansplatting.scene.vanilla_gaussian_model import (
    GaussianModel as VanillaGaussianModel,
)

from gaussiansplatting.arguments import (
    PipelineParams,
    OptimizationParams,
)
from omegaconf import OmegaConf

from argparse import ArgumentParser
from threestudio.utils.misc import (
    get_device,
    step_check,
    dilate_mask,
    erode_mask,
    fill_closed_areas,
)
from threestudio.utils.sam import LangSAMTextSegmentor, SceneSplatSegmentor
from threestudio.utils.camera import camera_ray_sample_points, project, unproject

from gaussiansplatting.scene.camera_scene import CamScene
import math
from GUI.EditGuidance import EditGuidance
from GUI.DelGuidance import DelGuidance

import os
import random
import ui_utils

import datetime
import subprocess
from pathlib import Path
from threestudio.utils.transform import (
    rotate_gaussians,
    translate_gaussians,
    scale_gaussians,
    default_model_mtx,
)


def generate_indoor_cameras(bbox_min, bbox_max, vertical_axis='Z',
                           grid_size=3, image_height=512, image_width=512):
    """
    Generate cameras for indoor scenes placed inside the bounding box.
    Cameras are positioned at human eye height and look in 8 directions.

    Args:
        bbox_min: (3,) array with [x_min, y_min, z_min]
        bbox_max: (3,) array with [x_max, y_max, z_max]
        vertical_axis: 'X', 'Y', or 'Z' - which axis is vertical
        grid_size: number of grid positions per horizontal dimension (3 = 3x3 grid)
        image_height: int
        image_width: int

    Returns:
        List of Simple_Camera objects
    """
    cameras = []
    fovy = math.radians(49.1)
    fovx = 2 * math.atan(math.tan(fovy / 2) * (image_width / image_height))

    # Determine which axes are horizontal and which is vertical
    axes = {'X': 0, 'Y': 1, 'Z': 2}
    vert_idx = axes[vertical_axis]
    horiz_indices = [i for i in range(3) if i != vert_idx]

    # Camera height: 40% up from bottom (human eye height)
    camera_height = bbox_min[vert_idx] + 0.4 * (bbox_max[vert_idx] - bbox_min[vert_idx])

    # Create grid positions in horizontal plane
    # Use inner 80% of space to avoid cameras too close to walls
    margin = 0.1
    h1_min = bbox_min[horiz_indices[0]] + margin * (bbox_max[horiz_indices[0]] - bbox_min[horiz_indices[0]])
    h1_max = bbox_max[horiz_indices[0]] - margin * (bbox_max[horiz_indices[0]] - bbox_min[horiz_indices[0]])
    h2_min = bbox_min[horiz_indices[1]] + margin * (bbox_max[horiz_indices[1]] - bbox_min[horiz_indices[1]])
    h2_max = bbox_max[horiz_indices[1]] - margin * (bbox_max[horiz_indices[1]] - bbox_min[horiz_indices[1]])

    h1_positions = np.linspace(h1_min, h1_max, grid_size)
    h2_positions = np.linspace(h2_min, h2_max, grid_size)

    # 8 viewing directions (N, NE, E, SE, S, SW, W, NW)
    # These are angles in the horizontal plane
    directions = [0, 45, 90, 135, 180, 225, 270, 315]  # degrees

    cam_id = 0
    for h1 in h1_positions:
        for h2 in h2_positions:
            # Build camera position
            cam_pos = np.zeros(3)
            cam_pos[horiz_indices[0]] = h1
            cam_pos[horiz_indices[1]] = h2
            cam_pos[vert_idx] = camera_height

            for angle_deg in directions:
                angle_rad = math.radians(angle_deg)

                # Look direction in horizontal plane
                look_h1 = math.cos(angle_rad)
                look_h2 = math.sin(angle_rad)

                # Build look-at point (1 meter away in that direction)
                lookat = cam_pos.copy()
                lookat[horiz_indices[0]] += look_h1
                lookat[horiz_indices[1]] += look_h2

                # Camera z-axis points toward lookat
                z_axis = lookat - cam_pos
                z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

                # World up vector (pointing up along vertical axis)
                world_up = np.zeros(3)
                if vertical_axis == 'Z':
                    world_up[2] = 1.0  # Z up
                elif vertical_axis == 'Y':
                    world_up[1] = -1.0  # Y down (negative Y is up in this convention)
                else:  # X
                    world_up[0] = 1.0

                # Handle gimbal lock
                if abs(np.dot(z_axis, world_up)) > 0.99:
                    world_up = np.array([1.0, 0.0, 0.0]) if vertical_axis != 'X' else np.array([0.0, 1.0, 0.0])

                x_axis = np.cross(world_up, z_axis)
                x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
                y_axis = np.cross(z_axis, x_axis)
                y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

                R = np.stack([x_axis, y_axis, z_axis], axis=0)
                T = -R @ cam_pos

                cam = Simple_Camera(
                    cam_id, R, T, fovx, fovy,
                    image_height, image_width, "", 0
                )
                cameras.append(cam)
                cam_id += 1

    return cameras


def generate_orbit_cameras_fibonacci(center, radius, n_cameras=100,
                                     image_height=512, image_width=512,
                                     hemisphere_only=False):
    """
    Generate cameras uniformly distributed on a sphere using Fibonacci spiral sampling.
    This provides much better coverage than fixed elevation angles.

    Args:
        center: (3,) numpy array, the center of the scene
        radius: float, distance from camera to center
        n_cameras: int, total number of cameras to generate
        image_height: int
        image_width: int
        hemisphere_only: bool, if True, only sample upper hemisphere (for ground objects)

    Returns:
        List of Simple_Camera objects
    """
    cameras = []
    fovy = math.radians(49.1)
    fovx = 2 * math.atan(math.tan(fovy / 2) * (image_width / image_height))

    # Fibonacci sphere sampling
    golden_ratio = (1 + math.sqrt(5)) / 2

    cam_id = 0
    for i in range(n_cameras * 2 if hemisphere_only else n_cameras):
        # Fibonacci sphere formula
        theta = 2 * math.pi * i / golden_ratio  # azimuth
        phi = math.acos(1 - 2 * (i + 0.5) / (n_cameras * 2 if hemisphere_only else n_cameras))  # polar angle

        # Skip lower hemisphere if hemisphere_only
        if hemisphere_only and phi > math.pi / 2:
            continue

        # Convert spherical to Cartesian (y-down convention to match original code)
        cam_x = center[0] + radius * math.sin(phi) * math.cos(theta)
        cam_y = center[1] - radius * math.cos(phi)  # negative y to match original convention
        cam_z = center[2] + radius * math.sin(phi) * math.sin(theta)
        cam_pos = np.array([cam_x, cam_y, cam_z])

        # Build rotation: z-axis points from cam toward center
        z_axis = center - cam_pos
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

        # Choose up vector carefully to avoid gimbal lock (match original convention)
        world_up = np.array([0.0, -1.0, 0.0])
        if abs(np.dot(z_axis, world_up)) > 0.99:
            world_up = np.array([1.0, 0.0, 0.0])

        x_axis = np.cross(world_up, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

        R = np.stack([x_axis, y_axis, z_axis], axis=0)
        T = -R @ cam_pos

        cam = Simple_Camera(
            cam_id, R, T, fovx, fovy,
            image_height, image_width, "", 0
        )
        cameras.append(cam)
        cam_id += 1

        if hemisphere_only and cam_id >= n_cameras:
            break

    return cameras


@torch.no_grad()
def score_cameras(candidates, gaussian, pipe, bg, mask_3d=None):
    """
    Score candidate cameras. Three hard filters applied in order:
      1. Camera must NOT be inside the point cloud (KNN proximity check)
      2. Rendered image must NOT be blurry (Laplacian variance check)
      3. Camera must see the target object (mask visibility check)
    Cameras failing any filter get score = -1.
    """
    import torch.nn.functional as F_fn
    from gaussiansplatting.gaussian_renderer import render as gs_render

    # Pre-compute KNN structure for inside-point-cloud detection
    all_xyz = gaussian.get_xyz.detach()  # [N, 3] on cuda
    # Sample a subset for efficiency if point cloud is large
    if all_xyz.shape[0] > 50000:
        indices = torch.randperm(all_xyz.shape[0], device=all_xyz.device)[:50000]
        xyz_sample = all_xyz[indices]
    else:
        xyz_sample = all_xyz
    # Compute typical inter-point distance (median of KNN-1 distances)
    # If camera is closer than this to many points, it's inside the cloud
    from torch import cdist
    # Use a random subset of 1000 points to estimate typical spacing
    n_est = min(1000, xyz_sample.shape[0])
    est_idx = torch.randperm(xyz_sample.shape[0], device=xyz_sample.device)[:n_est]
    est_pts = xyz_sample[est_idx]
    est_dists = cdist(est_pts, xyz_sample)  # [n_est, N_sample]
    # For each point, find distance to 10th nearest neighbor
    knn_dists, _ = est_dists.topk(11, dim=1, largest=False)  # include self
    typical_spacing = knn_dists[:, 10].median().item()  # 10th NN median distance
    min_cam_dist = typical_spacing * 3.0  # camera must be at least this far from dense regions
    print(f"[score_cameras] typical_spacing={typical_spacing:.4f}, min_cam_dist={min_cam_dist:.4f}")

    scores = []
    for i, cam in enumerate(candidates):
        # --- Filter 1: Check if camera is inside the point cloud ---
        cam_pos = torch.tensor(-cam.R.T @ cam.T, dtype=torch.float32, device='cuda')
        dists_to_cloud = torch.norm(xyz_sample - cam_pos.unsqueeze(0), dim=1)
        # Count how many points are very close to the camera
        n_close = (dists_to_cloud < min_cam_dist).sum().item()
        close_ratio = n_close / xyz_sample.shape[0]
        # If more than 5% of points are within min_cam_dist, camera is inside geometry
        if close_ratio > 0.05:
            scores.append((i, -1.0))
            continue

        # --- Filter 2: Render and check sharpness (Laplacian variance) ---
        pkg = gs_render(cam, gaussian, pipe, bg)
        img = pkg["render"]  # [3, H, W]
        gray = img.mean(dim=0)  # [H, W]

        gray_u = (gray * 255).unsqueeze(0).unsqueeze(0)
        lap_kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=torch.float32, device=img.device
        ).unsqueeze(0).unsqueeze(0)
        lap = F_fn.conv2d(gray_u, lap_kernel, padding=1)
        sharpness = lap.var().item()
        # Blurry images have very low Laplacian variance
        if sharpness < 500.0:
            scores.append((i, -1.0))
            continue

        # --- Filter 3: Must see the target object ---
        mask_ratio = 0.0
        if mask_3d is not None and mask_3d.sum() > 0:
            override = mask_3d.view(-1)[..., None].float().repeat(1, 3)
            sem = gs_render(cam, gaussian, pipe, bg, override_color=override)["render"]
            mask_2d = torch.norm(sem, dim=0)
            mask_ratio = (mask_2d > 0).float().mean().item()

        if mask_ratio < 0.005:
            scores.append((i, -1.0))
            continue

        # --- Score: target visibility (dominant) + sharpness + valid pixels ---
        valid_ratio = (gray > 0.01).float().mean().item()
        target_score = min(mask_ratio, 0.5)  # cap to avoid too-close views
        # Normalize sharpness to [0, 1] range (log scale)
        sharp_norm = min(math.log1p(sharpness) / 10.0, 1.0)

        score = target_score * 0.5 + sharp_norm * 0.3 + valid_ratio * 0.2
        scores.append((i, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def generate_adaptive_cameras(target_center, target_radius, scene_center,
                              n_cameras=80, image_height=512, image_width=512,
                              distance_factor=2.5):
    """
    Generate cameras adaptively focused on a target object.
    Cameras look at the target center from various angles.

    Args:
        target_center: (3,) numpy array, center of the target object
        target_radius: float, radius of the target object's bounding sphere
        scene_center: (3,) numpy array, center of the entire scene (for hemisphere detection)
        n_cameras: int, number of cameras to generate
        image_height: int
        image_width: int
        distance_factor: float, camera distance = target_radius * distance_factor

    Returns:
        List of Simple_Camera objects
    """
    # Determine if target is on the ground (use hemisphere sampling)
    # If target y is close to scene bottom, use hemisphere
    target_y = target_center[1]
    scene_y = scene_center[1]

    # If target is in lower half of scene, assume it's on ground -> use hemisphere
    hemisphere_only = target_y > scene_y

    camera_distance = target_radius * distance_factor

    print(f"[Adaptive Camera] Target center: {target_center}, radius: {target_radius:.3f}")
    print(f"[Adaptive Camera] Camera distance: {camera_distance:.3f}, hemisphere_only: {hemisphere_only}")

    return generate_orbit_cameras_fibonacci(
        center=target_center,
        radius=camera_distance,
        n_cameras=n_cameras,
        image_height=image_height,
        image_width=image_width,
        hemisphere_only=hemisphere_only
    )


def generate_orbit_cameras(center, radius, image_height=512, image_width=512,
                            n_azimuth=12, elevations=(-15, 0, 15, 30)):
    """
    Generate synthetic orbit cameras around a center point (legacy method).
    Returns a list of Simple_Camera objects.

    Args:
        center: (3,) numpy array, the center of the scene
        radius: float, distance from camera to center
        image_height: int
        image_width: int
        n_azimuth: int, number of azimuth angles per elevation
        elevations: list of elevation angles in degrees
    """
    cameras = []
    fovy = math.radians(49.1)  # ~50 degree FoV, reasonable default
    fovx = 2 * math.atan(math.tan(fovy / 2) * (image_width / image_height))

    cam_id = 0
    for elev_deg in elevations:
        elev = math.radians(elev_deg)
        for i in range(n_azimuth):
            azim = math.radians(360.0 * i / n_azimuth)

            # Camera position in world space
            cam_x = center[0] + radius * math.cos(elev) * math.sin(azim)
            cam_y = center[1] - radius * math.sin(elev)   # y-up convention
            cam_z = center[2] + radius * math.cos(elev) * math.cos(azim)
            cam_pos = np.array([cam_x, cam_y, cam_z])

            # Build rotation: z-axis points from cam toward center (look-at)
            z_axis = center - cam_pos
            z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-8)

            # world up vector; if camera is near the poles, use a different up
            world_up = np.array([0.0, -1.0, 0.0])
            if abs(np.dot(z_axis, world_up)) > 0.99:
                world_up = np.array([0.0, 0.0, 1.0])

            x_axis = np.cross(world_up, z_axis)
            x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-8)
            y_axis = np.cross(z_axis, x_axis)
            y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-8)

            # R: columns are x, y, z axes (camera-to-world rotation)
            # Simple_Camera expects R in colmap convention:
            # R is world-to-camera rotation (row vectors = camera axes in world)
            R = np.stack([x_axis, y_axis, z_axis], axis=0)  # (3,3) world-to-cam

            # T = -R @ cam_pos  (translation in camera space)
            T = -R @ cam_pos

            cam = Simple_Camera(
                cam_id, R, T, fovx, fovy,
                image_height, image_width, "", 0
            )
            cameras.append(cam)
            cam_id += 1

    return cameras


class WebUI:
    def __init__(self, cfg) -> None:
        self.gs_source = cfg.gs_source
        self.colmap_dir = cfg.colmap_dir  # may be None
        self.port = 8084
        # training cfg

        self.use_sam = False
        self.guidance = None
        self.stop_training = False
        self.inpaint_end_flag = False
        self.scale_depth = True
        self.depth_end_flag = False
        self.seg_scale = True
        self.seg_scale_end = False
        # from original system
        self.points3d = []
        self.gaussian = GaussianModel(
            sh_degree=0,
            anchor_weight_init_g0=1.0,
            anchor_weight_init=0.1,
            anchor_weight_multiplier=2,
        )
        # load
        self.gaussian.load_ply(self.gs_source)
        self.gaussian.max_radii2D = torch.zeros(
            (self.gaussian.get_xyz.shape[0]), device="cuda"
        )
        # front end related
        self.colmap_cameras = None
        self.render_cameras = None

        # diffusion model
        self.ip2p = None
        self.ctn_ip2p = None

        self.ctn_inpaint = None
        self.ctn_ip2p = None
        self.training = False

        if self.colmap_dir is not None:
            # Use real colmap cameras
            scene = CamScene(self.colmap_dir, h=512, w=512)
            self.cameras_extent = scene.cameras_extent
            self.colmap_cameras = scene.cameras
            print(f"[WebUI] Loaded {len(self.colmap_cameras)} colmap cameras.")
            # Store scene center for initial camera positioning
            self.scene_center = None
            self.scene_radius = None
        else:
            # No colmap: generate orbit cameras from point cloud
            print("[WebUI] No colmap_dir provided. Generating orbit cameras from point cloud...")
            xyz = self.gaussian.get_xyz.detach().cpu().numpy()
            center = xyz.mean(axis=0)
            # Use 90th percentile distance as radius to avoid outliers
            dists = np.linalg.norm(xyz - center, axis=1)
            radius = float(np.quantile(dists, 0.9)) * 0.3
            # cameras_extent: rough scene scale used by the optimizer
            self.cameras_extent = float(np.quantile(dists, 0.9))

            # Use Fibonacci sphere sampling for uniform coverage
            self.colmap_cameras = generate_orbit_cameras_fibonacci(
                center=center,
                radius=radius,
                n_cameras=100,  # 均匀分布的100个视角
                image_height=512,
                image_width=512,
            )
            print(f"[WebUI] Generated {len(self.colmap_cameras)} orbit cameras using Fibonacci sampling. "
                  f"Center={center}, Radius={radius:.3f}, cameras_extent={self.cameras_extent:.3f}")
            # Store scene center and radius for initial camera positioning
            self.scene_center = center
            self.scene_radius = radius

        self.background_tensor = torch.tensor(
            [0, 0, 0], dtype=torch.float32, device="cuda"
        )
        self.edit_frames = {}
        self.origin_frames = {}
        self.masks_2D = {}
        self.text_segmentor = LangSAMTextSegmentor().to(get_device())
        self.sam_predictor = self.text_segmentor.model.sam.predictor
        self.sam_predictor.is_image_set = True
        self.sam_features = {}
        self.scenesplat_segmentor = None  # lazy init when needed
        self.semantic_gauassian_masks = {}
        self.semantic_gauassian_masks["ALL"] = torch.ones_like(self.gaussian._opacity)

        # 后台预加载 SceneSplat 模型，不阻塞 UI
        def _preload_scenesplat():
            try:
                ckpt_path = "/workspace/GaussianEditor/ckpts/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth"
                self.scenesplat_segmentor = SceneSplatSegmentor(
                    checkpoint_path=ckpt_path, device="cuda:0"
                )
                self.scenesplat_segmentor._ckpt_path = ckpt_path
                print("[SceneSplat] 模型预加载完成")
            except Exception as e:
                print(f"[SceneSplat] 模型预加载失败: {e}")
        threading.Thread(target=_preload_scenesplat, daemon=True).start()

        self.parser = ArgumentParser(description="Training script parameters")
        self.pipe = PipelineParams(self.parser)

        # status
        self.display_semantic_mask = False
        self.display_point_prompt = False

        self.viewer_need_update = False
        self.system_need_update = False
        self.inpaint_again = True
        self.scale_depth = True

        self.server = viser.ViserServer(port=self.port)
        self.add_theme()

        # Set initial camera position to look at scene center
        if hasattr(self, 'scene_center') and self.scene_center is not None:
            # For generated orbit cameras, set initial view
            initial_distance = self.scene_radius * 1.5  # slightly further than the orbit cameras
            initial_position = self.scene_center + np.array([initial_distance, 0, 0])

            # Set up callback to position camera when client connects
            @self.server.on_client_connect
            def _(client):
                client.camera.position = initial_position
                client.camera.look_at = self.scene_center

        self.draw_flag = True
        with self.server.add_gui_folder("Render Setting"):
            self.resolution_slider = self.server.add_gui_slider(
                "Resolution", min=384, max=4096, step=2, initial_value=2048
            )

            self.FoV_slider = self.server.add_gui_slider(
                "FoV Scaler", min=0.2, max=2, step=0.1, initial_value=1
            )

            self.fps = self.server.add_gui_text(
                "FPS", initial_value="-1", disabled=True
            )
            self.renderer_output = self.server.add_gui_dropdown(
                "Renderer Output",
                [
                    "comp_rgb",
                ],
            )
            self.save_button = self.server.add_gui_button("Save Gaussian")

            self.frame_show = self.server.add_gui_checkbox(
                "Show Frame", initial_value=False
            )

        with self.server.add_gui_folder("Semantic Tracing"):
            self.sam_enabled = self.server.add_gui_checkbox(
                "Enable SAM",
                initial_value=False,
            )
            self.add_sam_points = self.server.add_gui_checkbox(
                "Add SAM Points", initial_value=False
            )
            self.sam_group_name = self.server.add_gui_text(
                "SAM Group Name", initial_value="table"
            )
            self.clear_sam_pins = self.server.add_gui_button(
                "Clear SAM Pins",
            )
            self.text_seg_prompt = self.server.add_gui_text(
                "Text Seg Prompt", initial_value="a bike"
            )
            self.use_scenesplat_checkbox = self.server.add_gui_checkbox(
                "Use SceneSplat 3D Seg", initial_value=False
            )
            self.scenesplat_ckpt_path = self.server.add_gui_text(
                "SceneSplat Checkpoint", initial_value="/workspace/GaussianEditor/ckpts/lang-pretrain-concat-scan-ppv2-matt-mcmc-wo-normal-contrastive.pth"
            )
            self.semantic_groups = self.server.add_gui_dropdown(
                "Semantic Group",
                options=["ALL"],
            )

            self.seg_cam_num = self.server.add_gui_slider(
                "Seg Camera Nums", min=6, max=200, step=1, initial_value=24
            )

            self.mask_thres = self.server.add_gui_slider(
                "Seg Threshold", min=0.2, max=0.99999, step=0.00001, initial_value=0.7, visible=False
            )
            self.seg_status = self.server.add_gui_text(
                "Seg Status", initial_value="", disabled=True, visible=False
            )

            self.show_semantic_mask = self.server.add_gui_checkbox(
                "Show Semantic Mask", initial_value=False
            )
            self.seg_scale_end_button = self.server.add_gui_button(
                "End Seg Scale!",
                visible=False,
            )
            self.submit_seg_prompt = self.server.add_gui_button("Tracing Begin!")

        with self.server.add_gui_folder("Edit Setting"):
            self.edit_type = self.server.add_gui_dropdown(
                "Edit Type", ("Edit", "Delete", "Add")
            )
            self.guidance_type = self.server.add_gui_dropdown(
                "Guidance Type", ("InstructPix2Pix", "ControlNet-Pix2Pix")
            )
            self.edit_frame_show = self.server.add_gui_checkbox(
                "Show Edit Frame", initial_value=True, visible=False
            )
            self.edit_text = self.server.add_gui_text(
                "Text",
                initial_value="",
                visible=True,
            )
            self.draw_bbox = self.server.add_gui_checkbox(
                "Draw Bounding Box", initial_value=False, visible=False
            )
            self.left_up = self.server.add_gui_vector2(
                "Left UP",
                initial_value=(0, 0),
                step=1,
                visible=False,
            )
            self.right_down = self.server.add_gui_vector2(
                "Right Down",
                initial_value=(0, 0),
                step=1,
                visible=False,
            )

            self.inpaint_seed = self.server.add_gui_slider(
                "Inpaint Seed", min=0, max=1000, step=1, initial_value=0, visible=False
            )

            self.refine_text = self.server.add_gui_text(
                "Refine Text",
                initial_value="",
                visible=False,
            )
            self.inpaint_end = self.server.add_gui_button(
                "End 2D Inpainting!",
                visible=False,
            )

            self.depth_scaler = self.server.add_gui_slider(
                "Depth Scale", min=0.0, max=5.0, step=0.01, initial_value=1.0, visible=False
            )
            self.depth_end = self.server.add_gui_button(
                "End Depth Scale!",
                visible=False,
            )
            self.edit_begin_button = self.server.add_gui_button("Edit Begin!")
            self.edit_end_button = self.server.add_gui_button(
                "End Editing!", visible=False
            )

            # ── Manual camera capture for editing ──
            # User browses the viewer to find clear views of the target,
            # clicks "Capture View" to save each one. These are used as
            # edit cameras instead of auto-generated ones.
            self.capture_view_button = self.server.add_gui_button("Capture View")
            self.clear_captures_button = self.server.add_gui_button("Clear Captures")
            self.capture_count_text = self.server.add_gui_text(
                "Captured Views", initial_value="0", disabled=True
            )
            self.captured_cameras = []  # list of Simple_Camera

            with self.server.add_gui_folder("Advanced Options"):
                self.edit_cam_num = self.server.add_gui_slider(
                    "Camera Num", min=12, max=200, step=1, initial_value=16
                )
                self.edit_train_steps = self.server.add_gui_slider(
                    "Total Step", min=0, max=5000, step=100, initial_value=1500
                )
                self.densify_until_step = self.server.add_gui_slider(
                    "Densify Until Step",
                    min=0,
                    max=5000,
                    step=50,
                    initial_value=1300,
                )

                self.densification_interval = self.server.add_gui_slider(
                    "Densify Interval",
                    min=25,
                    max=1000,
                    step=25,
                    initial_value=100,
                )
                self.max_densify_percent = self.server.add_gui_slider(
                    "Max Densify Percent",
                    min=0.0,
                    max=1.0,
                    step=0.001,
                    initial_value=0.01,
                )
                self.min_opacity = self.server.add_gui_slider(
                    "Min Opacity",
                    min=0.0,
                    max=0.1,
                    step=0.0001,
                    initial_value=0.005,
                )

                self.per_editing_step = self.server.add_gui_slider(
                    "Edit Interval", min=4, max=48, step=1, initial_value=10
                )
                self.edit_begin_step = self.server.add_gui_slider(
                    "Edit Begin Step", min=0, max=5000, step=100, initial_value=0
                )
                self.edit_until_step = self.server.add_gui_slider(
                    "Edit Until Step", min=0, max=5000, step=100, initial_value=1000
                )

                self.inpaint_scale = self.server.add_gui_slider(
                    "Inpaint Scale", min=0.1, max=10, step=0.1, initial_value=1, visible=False
                )

                self.mask_dilate = self.server.add_gui_slider(
                    "Mask Dilate", min=1, max=30, step=1, initial_value=15, visible=False
                )
                self.fix_holes = self.server.add_gui_checkbox(
                    "Fix Holes", initial_value=True, visible=False
                )
                with self.server.add_gui_folder("Learning Rate Scaler"):
                    self.gs_lr_scaler = self.server.add_gui_slider(
                        "XYZ LR Init", min=0.0, max=10.0, step=0.1, initial_value=3.0
                    )
                    self.gs_lr_end_scaler = self.server.add_gui_slider(
                        "XYZ LR End", min=0.0, max=10.0, step=0.1, initial_value=2.0
                    )
                    self.color_lr_scaler = self.server.add_gui_slider(
                        "Color LR", min=0.0, max=10.0, step=0.1, initial_value=3.0
                    )
                    self.opacity_lr_scaler = self.server.add_gui_slider(
                        "Opacity LR", min=0.0, max=10.0, step=0.1, initial_value=2.0
                    )
                    self.scaling_lr_scaler = self.server.add_gui_slider(
                        "Scale LR", min=0.0, max=10.0, step=0.1, initial_value=2.0
                    )
                    self.rotation_lr_scaler = self.server.add_gui_slider(
                        "Rotation LR", min=0.0, max=10.0, step=0.1, initial_value=2.0
                    )

                with self.server.add_gui_folder("Loss Options"):
                    self.lambda_l1 = self.server.add_gui_slider(
                        "Lambda L1", min=0, max=100, step=1, initial_value=10
                    )
                    self.lambda_p = self.server.add_gui_slider(
                        "Lambda Perceptual", min=0, max=100, step=1, initial_value=10
                    )

                    self.use_multiview_consistency = self.server.add_gui_checkbox(
                        "Multi-view Consistency", initial_value=False
                    )
                    self.lambda_consistency = self.server.add_gui_slider(
                        "Consistency Weight", min=0.0, max=20.0, step=0.5, initial_value=5.0,
                        visible=False,
                    )

                    self.pre_generate_edits = self.server.add_gui_checkbox(
                        "Pre-Generate Edits (Fast)", initial_value=True,
                    )

                    self.anchor_weight_init_g0 = self.server.add_gui_slider(
                        "Anchor Init G0", min=0., max=10., step=0.05, initial_value=0.05
                    )
                    self.anchor_weight_init = self.server.add_gui_slider(
                        "Anchor Init", min=0., max=10., step=0.05, initial_value=0.1
                    )
                    self.anchor_weight_multiplier = self.server.add_gui_slider(
                        "Anchor Multiplier", min=1., max=10., step=0.1, initial_value=1.3
                    )

                    self.lambda_anchor_color = self.server.add_gui_slider(
                        "Lambda Anchor Color", min=0, max=500, step=1, initial_value=0
                    )
                    self.lambda_anchor_geo = self.server.add_gui_slider(
                        "Lambda Anchor Geo", min=0, max=500, step=1, initial_value=50
                    )
                    self.lambda_anchor_scale = self.server.add_gui_slider(
                        "Lambda Anchor Scale", min=0, max=500, step=1, initial_value=50
                    )
                    self.lambda_anchor_opacity = self.server.add_gui_slider(
                        "Lambda Anchor Opacity", min=0, max=500, step=1, initial_value=50
                    )
                    self.anchor_term = [self.anchor_weight_init_g0, self.anchor_weight_init,
                                        self.anchor_weight_multiplier,
                                        self.lambda_anchor_color, self.lambda_anchor_geo,
                                        self.lambda_anchor_scale, self.lambda_anchor_opacity, ]

        @self.inpaint_seed.on_update
        def _(_):
            self.inpaint_again = True

        @self.depth_scaler.on_update
        def _(_):
            self.scale_depth = True

        @self.use_multiview_consistency.on_update
        def _(_):
            self.lambda_consistency.visible = self.use_multiview_consistency.value

        @self.mask_thres.on_update
        def _(_):
            self.seg_scale = True

        @self.use_scenesplat_checkbox.on_update
        def _(_):
            if self.use_scenesplat_checkbox.value:
                self.mask_thres.visible = True
                self.seg_status.visible = True
                if self.scenesplat_segmentor is None:
                    self.seg_status.value = "模型加载中，请稍等..."
                else:
                    self.seg_status.value = "模型已就绪，可以开始分割"
            else:
                self.mask_thres.visible = False
                self.seg_status.visible = False

        @self.edit_type.on_update
        def _(_):
            if self.edit_type.value == "Edit":
                self.edit_text.visible = True
                self.refine_text.visible = False
                for term in self.anchor_term:
                    term.visible = True
                self.inpaint_scale.visible = False
                self.mask_dilate.visible = False
                self.fix_holes.visible = False
                self.per_editing_step.visible = True
                self.edit_begin_step.visible = True
                self.edit_until_step.visible = True
                self.draw_bbox.visible = False
                self.left_up.visible = False
                self.right_down.visible = False
                self.inpaint_seed.visible = False
                self.inpaint_end.visible = False
                self.depth_scaler.visible = False
                self.depth_end.visible = False
                self.edit_frame_show.visible = True
                self.guidance_type.visible = True

            elif self.edit_type.value == "Delete":
                self.edit_text.visible = True
                self.refine_text.visible = False
                for term in self.anchor_term:
                    term.visible = True
                self.inpaint_scale.visible = True
                self.mask_dilate.visible = True
                self.fix_holes.visible = True
                self.edit_cam_num.value = 24
                self.densification_interval.value = 50
                self.per_editing_step.visible = False
                self.edit_begin_step.visible = False
                self.edit_until_step.visible = False
                self.draw_bbox.visible = False
                self.left_up.visible = False
                self.right_down.visible = False
                self.inpaint_seed.visible = False
                self.inpaint_end.visible = False
                self.depth_scaler.visible = False
                self.depth_end.visible = False
                self.edit_frame_show.visible = True
                self.guidance_type.visible = False

            elif self.edit_type.value == "Add":
                self.edit_text.visible = True
                self.refine_text.visible = False
                for term in self.anchor_term:
                    term.visible = False
                self.inpaint_scale.visible = False
                self.mask_dilate.visible = False
                self.fix_holes.visible = False
                self.per_editing_step.visible = True
                self.edit_begin_step.visible = True
                self.edit_until_step.visible = True
                self.draw_bbox.visible = True
                self.left_up.visible = True
                self.right_down.visible = True
                self.inpaint_seed.visible = False
                self.inpaint_end.visible = False
                self.depth_scaler.visible = False
                self.depth_end.visible = False
                self.edit_frame_show.visible = False
                self.guidance_type.visible = False

        @self.save_button.on_click
        def _(_):
            current_time = datetime.datetime.now()
            formatted_time = current_time.strftime("%Y-%m-%d-%H:%M")
            self.gaussian.save_ply(os.path.join("ui_result", "{}.ply".format(formatted_time)))

        @self.inpaint_end.on_click
        def _(_):
            self.inpaint_end_flag = True

        @self.seg_scale_end_button.on_click
        def _(_):
            self.seg_scale_end = True

        @self.depth_end.on_click
        def _(_):
            self.depth_end_flag = True

        @self.edit_end_button.on_click
        def _(event: viser.GuiEvent):
            self.stop_training = True

        @self.capture_view_button.on_click
        def _(event: viser.GuiEvent):
            """Capture the current viewer camera as an edit camera."""
            clients = list(self.server.get_clients().values())
            if len(clients) == 0:
                print("[Capture] No client connected!")
                return
            viser_cam = clients[0].camera
            R = tf.SO3(viser_cam.wxyz).as_matrix()
            T_vec = -R.T @ viser_cam.position
            fovy = viser_cam.fov * self.FoV_slider.value
            if not hasattr(self, 'aspect'):
                self.aspect = viser_cam.aspect
            fovx = 2 * math.atan(math.tan(fovy / 2) * self.aspect)
            cam = Simple_Camera(
                len(self.captured_cameras), R, T_vec, fovx, fovy,
                512, 512, "", 0
            )
            self.captured_cameras.append(cam)
            self.capture_count_text.value = str(len(self.captured_cameras))
            print(f"[Capture] View #{len(self.captured_cameras)} captured. "
                  f"Position: {(-R.T @ T_vec).round(3)}")

        @self.clear_captures_button.on_click
        def _(_):
            self.captured_cameras.clear()
            self.capture_count_text.value = "0"
            print("[Capture] All captured views cleared.")

        @self.edit_begin_button.on_click
        def _(event: viser.GuiEvent):
            self.edit_begin_button.visible = False
            self.edit_end_button.visible = True
            if self.training:
                return
            self.training = True
            self.configure_optimizers()
            self.gaussian.update_anchor_term(
                anchor_weight_init_g0=self.anchor_weight_init_g0.value,
                anchor_weight_init=self.anchor_weight_init.value,
                anchor_weight_multiplier=self.anchor_weight_multiplier.value,
            )

            if self.edit_type.value == "Add":
                self.add(self.camera)
            else:
                self.edit_frame_show.visible = True

                # ── Camera selection for editing ──
                # Priority: use manually captured cameras if available.
                # Otherwise fall back to auto-generation.
                if len(self.captured_cameras) > 0:
                    # Use manually captured cameras
                    print(f"[Edit] Using {len(self.captured_cameras)} manually captured cameras")
                    edit_cameras, train_frames, train_frustums = [], [], []
                    for cam_idx, cam in enumerate(self.captured_cameras):
                        position = -cam.R.T @ cam.T
                        wxyz = tf.SO3.from_matrix(cam.R.T).wxyz
                        T_wc = tf.SE3.from_rotation_and_translation(tf.SO3(wxyz), position)
                        wxyz = T_wc.rotation().wxyz
                        position = T_wc.translation()
                        frame = self.server.add_frame(
                            f"/train/frame_{cam_idx}", wxyz=wxyz, position=position,
                            axes_length=0.1, axes_radius=0.01)
                        H, W = cam.image_height, cam.image_width
                        frustum = self.server.add_camera_frustum(
                            f"/train/frame_{cam_idx}/frustum",
                            fov=cam.FoVy, aspect=W / H, scale=0.15, image=None)

                        def attach_callback(frustum, frame):
                            @frustum.on_click
                            def _(_):
                                for client in self.server.get_clients().values():
                                    client.camera.wxyz = frame.wxyz
                                    client.camera.position = frame.position
                        attach_callback(frustum, frame)

                        edit_cameras.append(cam)
                        train_frames.append(frame)
                        train_frustums.append(frustum)

                    # Save debug views
                    import os
                    from torchvision.utils import save_image
                    from gaussiansplatting.gaussian_renderer import render as gs_render
                    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_views")
                    os.makedirs(save_dir, exist_ok=True)
                    for idx, cam in enumerate(edit_cameras):
                        with torch.no_grad():
                            pkg = gs_render(cam, self.gaussian, self.pipe, self.background_tensor)
                            img = pkg["render"].clamp(0, 1)
                            save_image(img, os.path.join(save_dir, f"view_{idx:03d}.png"))
                    print(f"[Edit] Saved {len(edit_cameras)} captured views to {save_dir}")

                elif self.gaussian.mask is not None and self.gaussian.mask.sum() > 0:
                    mask = self.gaussian.mask
                    target_xyz = self.gaussian.get_xyz[mask.bool()].detach().cpu().numpy()
                    target_center = target_xyz.mean(axis=0)
                    target_dists = np.linalg.norm(target_xyz - target_center, axis=1)
                    target_radius = float(np.quantile(target_dists, 0.9))

                    from gaussiansplatting.gaussian_renderer import render as gs_render
                    import torch.nn.functional as F_fn

                    scene_center = self.scene_center if self.scene_center is not None else \
                        self.gaussian.get_xyz.detach().cpu().numpy().mean(axis=0)

                    print(f"[Edit] Target center={target_center.round(3)}, "
                          f"target_radius={target_radius:.3f}, "
                          f"cameras_extent={self.cameras_extent:.3f}")

                    # ── Step 1: Find safe directions from orbit cameras ──
                    # Render each orbit camera and check depth quality.
                    sobel_x = torch.tensor(
                        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                        dtype=torch.float32, device='cuda'
                    ).unsqueeze(0).unsqueeze(0)
                    sobel_y = torch.tensor(
                        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                        dtype=torch.float32, device='cuda'
                    ).unsqueeze(0).unsqueeze(0)

                    safe_directions = []  # unit vectors from scene_center to camera
                    print(f"[Edit] Probing {len(self.colmap_cameras)} orbit cameras for safe directions...")

                    # ── Diagnostic: save orbit camera renders + depth stats ──
                    import os
                    from torchvision.utils import save_image
                    diag_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_views", "diagnostics")
                    orbit_dir = os.path.join(diag_dir, "orbit_cameras")
                    os.makedirs(orbit_dir, exist_ok=True)

                    orbit_stats = []  # (idx, depth_valid_ratio, depth_grad_median, is_safe)

                    for ci, cam in enumerate(self.colmap_cameras):
                        cam_pos = -cam.R.T @ cam.T
                        pkg = gs_render(cam, self.gaussian, self.pipe, self.background_tensor)
                        img = pkg["render"]        # [3, H, W]
                        depth = pkg["depth_3dgs"]  # [1, H, W]
                        depth_sq = depth.squeeze(0)

                        # Check depth validity
                        depth_valid_ratio = (depth_sq > 0).float().mean().item()
                        if depth_valid_ratio < 0.7:
                            orbit_stats.append((ci, depth_valid_ratio, -1, False))
                            continue

                        # Check depth smoothness
                        d_valid = depth_sq[depth_sq > 0]
                        d_min, d_max = d_valid.min().item(), d_valid.max().item()
                        d_range = d_max - d_min
                        if d_range < 1e-6:
                            orbit_stats.append((ci, depth_valid_ratio, -1, False))
                            continue

                        depth_norm = ((depth_sq - d_min) / d_range).unsqueeze(0).unsqueeze(0)
                        gx = F_fn.conv2d(depth_norm, sobel_x, padding=1)
                        gy = F_fn.conv2d(depth_norm, sobel_y, padding=1)
                        grad_mag = torch.sqrt(gx ** 2 + gy ** 2).squeeze()
                        valid_grads = grad_mag[depth_sq > 0]
                        depth_grad_median = valid_grads.median().item()

                        is_safe = depth_grad_median < 0.02
                        orbit_stats.append((ci, depth_valid_ratio, depth_grad_median, is_safe))

                        if is_safe:
                            direction = cam_pos - scene_center
                            direction = direction / (np.linalg.norm(direction) + 1e-8)
                            safe_directions.append(direction)

                        # Save RGB + depth visualization for first 20 orbit cameras
                        if ci < 20:
                            save_image(img.clamp(0, 1), os.path.join(orbit_dir, f"orbit_{ci:03d}_rgb.png"))
                            # Normalize depth for visualization
                            d_vis = depth_sq.clone()
                            if d_vis.max() > 0:
                                d_vis = d_vis / d_vis.max()
                            save_image(d_vis.unsqueeze(0), os.path.join(orbit_dir, f"orbit_{ci:03d}_depth.png"))

                    # Print orbit stats summary
                    n_safe = sum(1 for s in orbit_stats if s[3])
                    grad_values = [s[2] for s in orbit_stats if s[2] >= 0]
                    if grad_values:
                        print(f"[Edit] Orbit depth_grad_median stats: "
                              f"min={min(grad_values):.4f}, max={max(grad_values):.4f}, "
                              f"mean={np.mean(grad_values):.4f}, median={np.median(grad_values):.4f}")
                    print(f"[Edit] Found {n_safe} safe directions out of {len(self.colmap_cameras)} orbit cameras")

                    if len(safe_directions) == 0:
                        # Fallback: use all orbit camera directions
                        print("[Edit] WARNING: No safe directions found, using all orbit directions")
                        for cam in self.colmap_cameras:
                            cam_pos = -cam.R.T @ cam.T
                            direction = cam_pos - scene_center
                            direction = direction / (np.linalg.norm(direction) + 1e-8)
                            safe_directions.append(direction)

                    # ── Step 2: Generate cameras along safe directions, centered on target ──
                    # For each safe direction, place cameras at several distances from target.
                    fovy = math.radians(49.1)
                    fovx = 2 * math.atan(math.tan(fovy / 2) * (512 / 512))

                    # Distance range: from close to far
                    scene_r = self.cameras_extent
                    dist_min = max(target_radius * 3.0, scene_r * 0.05)
                    dist_max = scene_r * 0.5
                    distances = np.geomspace(dist_min, dist_max, num=4)

                    print(f"[Edit] Generating cameras along {len(safe_directions)} safe directions "
                          f"at distances {[f'{d:.3f}' for d in distances]}")

                    candidate_cameras = []
                    for direction in safe_directions:
                        for dist in distances:
                            cam_pos = target_center + direction * dist

                            # Build rotation: look at target_center
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

                            cam = Simple_Camera(
                                len(candidate_cameras), R, T, fovx, fovy,
                                512, 512, "", 0
                            )
                            candidate_cameras.append(cam)

                    print(f"[Edit] Generated {len(candidate_cameras)} candidate cameras")

                    # ── Step 3: Filter by depth quality + target visibility ──
                    override = mask.view(-1)[..., None].float().repeat(1, 3)
                    cam_scores = []
                    n_bad_depth = 0
                    n_no_target = 0

                    # Diagnostic: save candidate renders
                    cand_dir = os.path.join(diag_dir, "candidates")
                    os.makedirs(cand_dir, exist_ok=True)
                    cand_pass_idx = 0

                    for i, cam in enumerate(candidate_cameras):
                        pkg = gs_render(cam, self.gaussian, self.pipe, self.background_tensor)
                        img = pkg["render"]
                        depth = pkg["depth_3dgs"]
                        depth_sq = depth.squeeze(0)

                        # Depth validity
                        depth_valid_ratio = (depth_sq > 0).float().mean().item()
                        if depth_valid_ratio < 0.6:
                            n_bad_depth += 1
                            continue

                        # Depth smoothness
                        d_valid = depth_sq[depth_sq > 0]
                        d_min, d_max = d_valid.min().item(), d_valid.max().item()
                        d_range = d_max - d_min
                        if d_range < 1e-6:
                            n_bad_depth += 1
                            continue

                        depth_norm = ((depth_sq - d_min) / d_range).unsqueeze(0).unsqueeze(0)
                        gx = F_fn.conv2d(depth_norm, sobel_x, padding=1)
                        gy = F_fn.conv2d(depth_norm, sobel_y, padding=1)
                        grad_mag = torch.sqrt(gx ** 2 + gy ** 2).squeeze()
                        valid_grads = grad_mag[depth_sq > 0]
                        depth_grad_median = valid_grads.median().item()

                        if depth_grad_median > 0.025:
                            n_bad_depth += 1
                            continue

                        # Target visibility
                        sem = gs_render(cam, self.gaussian, self.pipe,
                                        self.background_tensor,
                                        override_color=override)["render"]
                        mask_2d = torch.norm(sem, dim=0)
                        mask_ratio = (mask_2d > 0).float().mean().item()

                        if mask_ratio < 0.001:
                            n_no_target += 1
                            continue

                        # Save EVERY camera that passes all filters (RGB + depth + stats)
                        save_image(img.clamp(0, 1), os.path.join(cand_dir, f"pass_{cand_pass_idx:03d}_rgb.png"))
                        d_vis = depth_sq.clone()
                        if d_vis.max() > 0:
                            d_vis = d_vis / d_vis.max()
                        save_image(d_vis.unsqueeze(0), os.path.join(cand_dir, f"pass_{cand_pass_idx:03d}_depth.png"))
                        print(f"[Edit] Candidate {i} PASSED: depth_valid={depth_valid_ratio:.3f}, "
                              f"depth_grad_median={depth_grad_median:.4f}, mask_ratio={mask_ratio:.4f}")
                        cand_pass_idx += 1

                        # Score: depth quality + visibility
                        depth_quality = 1.0 - min(depth_grad_median / 0.025, 1.0)
                        vis_score = min(mask_ratio, 0.3)
                        score = depth_quality * 0.5 + vis_score * 0.4 + depth_valid_ratio * 0.1
                        cam_scores.append((i, score, mask_ratio, depth_grad_median))

                    print(f"[Edit] Filter results: {len(candidate_cameras)} total, "
                          f"{n_bad_depth} bad depth, {n_no_target} no target, "
                          f"{len(cam_scores)} passed")

                    # ── Step 4: Select with angular diversity ──
                    cam_scores.sort(key=lambda x: x[1], reverse=True)
                    n_want = self.edit_cam_num.value

                    if len(cam_scores) == 0:
                        print("[Edit] WARNING: No good cameras found! Falling back to random scene cameras.")
                        edit_cameras, train_frames, train_frustums = ui_utils.sample_train_camera(
                            self.colmap_cameras, self.edit_cam_num.value, self.server)
                    else:
                        selected_indices = []
                        selected_dirs = []
                        for idx, score, vis, dg in cam_scores:
                            if len(selected_indices) >= n_want:
                                break
                            cam = candidate_cameras[idx]
                            cam_pos = -cam.R.T @ cam.T
                            view_dir = target_center - cam_pos
                            view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)

                            too_close = False
                            for sd in selected_dirs:
                                if np.dot(view_dir, sd) > 0.95:
                                    too_close = True
                                    break
                            if too_close:
                                continue

                            selected_indices.append((idx, score, vis, dg))
                            selected_dirs.append(view_dir)

                        # Fill remaining if diversity was too strict
                        if len(selected_indices) < n_want:
                            existing = {s[0] for s in selected_indices}
                            for idx, score, vis, dg in cam_scores:
                                if len(selected_indices) >= n_want:
                                    break
                                if idx not in existing:
                                    selected_indices.append((idx, score, vis, dg))

                        best_cameras = [candidate_cameras[i] for i, _, _, _ in selected_indices]
                        print(f"[Edit] Selected {len(best_cameras)} cameras. "
                              f"Score: {selected_indices[0][1]:.4f} ~ {selected_indices[-1][1]:.4f}, "
                              f"Visibility: {selected_indices[0][2]:.4f} ~ {selected_indices[-1][2]:.4f}, "
                              f"Depth grad: {selected_indices[0][3]:.4f} ~ {selected_indices[-1][3]:.4f}")

                        edit_cameras, train_frames, train_frustums = [], [], []
                        for cam_idx, cam in enumerate(best_cameras):
                            position = -cam.R.T @ cam.T
                            wxyz = tf.SO3.from_matrix(cam.R.T).wxyz
                            T_wc = tf.SE3.from_rotation_and_translation(tf.SO3(wxyz), position)
                            wxyz = T_wc.rotation().wxyz
                            position = T_wc.translation()
                            frame = self.server.add_frame(
                                f"/train/frame_{cam_idx}", wxyz=wxyz, position=position,
                                axes_length=0.1, axes_radius=0.01)
                            H, W = cam.image_height, cam.image_width
                            frustum = self.server.add_camera_frustum(
                                f"/train/frame_{cam_idx}/frustum",
                                fov=cam.FoVy, aspect=W / H, scale=0.15, image=None)

                            def attach_callback(frustum, frame):
                                @frustum.on_click
                                def _(_):
                                    for client in self.server.get_clients().values():
                                        client.camera.wxyz = frame.wxyz
                                        client.camera.position = frame.position
                            attach_callback(frustum, frame)

                            edit_cameras.append(cam)
                            train_frames.append(frame)
                            train_frustums.append(frustum)
                else:
                    print("[Edit] No segmentation mask found, using random cameras.")
                    edit_cameras, train_frames, train_frustums = ui_utils.sample_train_camera(
                        self.colmap_cameras,
                        self.edit_cam_num.value,
                        self.server,
                    )
                if self.edit_type.value == "Edit":
                    self.edit(edit_cameras, train_frames, train_frustums)

                elif self.edit_type.value == "Delete":
                    self.delete(edit_cameras, train_frames, train_frustums)

                ui_utils.remove_all(train_frames)
                ui_utils.remove_all(train_frustums)
                self.edit_frame_show.visible = False
            self.guidance = None
            self.training = False
            self.gaussian.anchor_postfix()
            self.edit_begin_button.visible = True
            self.edit_end_button.visible = False

        @self.submit_seg_prompt.on_click
        def _(_):
            if not self.sam_enabled.value:
                text_prompt = self.text_seg_prompt.value
                print("[Segmentation Prompt]", text_prompt)
                _, semantic_gaussian_mask = self.update_mask(text_prompt)
            else:
                text_prompt = self.sam_group_name.value
                self.sam_enabled.value = False
                self.add_sam_points.value = False
                _, semantic_gaussian_mask = self.update_sam_mask_with_point_prompt(
                    save_mask=True
                )

            self.semantic_gauassian_masks[text_prompt] = semantic_gaussian_mask
            if text_prompt not in self.semantic_groups.options:
                self.semantic_groups.options += (text_prompt,)
            self.semantic_groups.value = text_prompt

        @self.semantic_groups.on_update
        def _(_):
            semantic_mask = self.semantic_gauassian_masks[self.semantic_groups.value]
            self.gaussian.set_mask(semantic_mask)
            self.gaussian.apply_grad_mask(semantic_mask)

        @self.edit_frame_show.on_update
        def _(_):
            if self.guidance is not None:
                for _ in self.guidance.train_frames:
                    _.visible = self.edit_frame_show.value
                for _ in self.guidance.train_frustums:
                    _.visible = self.edit_frame_show.value
                self.guidance.visible = self.edit_frame_show.value

        # Show camera pose frames (only when colmap cameras are real/meaningful)
        with torch.no_grad():
            self.frames = []
            random.seed(0)
            frame_index = random.sample(
                range(0, len(self.colmap_cameras)),
                min(len(self.colmap_cameras), 40),
            )
            for i in frame_index:
                self.make_one_camera_pose_frame(i)

        @self.frame_show.on_update
        def _(_):
            for frame in self.frames:
                frame.visible = self.frame_show.value
            self.server.world_axes.visible = self.frame_show.value

        @self.server.on_scene_click
        def _(pointer):
            self.click_cb(pointer)

        @self.clear_sam_pins.on_click
        def _(_):
            self.clear_points3d()

    def make_one_camera_pose_frame(self, idx):
        cam = self.colmap_cameras[idx]

        # Support both colmap cameras (with qvec) and Simple_Camera (with R matrix)
        if hasattr(cam, 'qvec') and cam.qvec is not None:
            T_world_camera = tf.SE3.from_rotation_and_translation(
                tf.SO3(cam.qvec), cam.T
            ).inverse()
        else:
            # Simple_Camera: R is world-to-cam, T is translation in cam space
            # world position = -R.T @ T
            position = -cam.R.T @ cam.T
            # R.T is cam-to-world rotation; convert to quaternion via SO3
            wxyz = tf.SO3.from_matrix(cam.R.T).wxyz
            T_world_camera = tf.SE3.from_rotation_and_translation(
                tf.SO3(wxyz), position
            )

        wxyz = T_world_camera.rotation().wxyz
        position = T_world_camera.translation()

        frame = self.server.add_frame(
            f"/colmap/frame_{idx}",
            wxyz=wxyz,
            position=position,
            axes_length=0.2,
            axes_radius=0.01,
            visible=False,
        )
        self.frames.append(frame)

        @frame.on_click
        def _(event: viser.GuiEvent):
            client = event.client
            assert client is not None
            T_world_current = tf.SE3.from_rotation_and_translation(
                tf.SO3(client.camera.wxyz), client.camera.position
            )

            T_world_target = tf.SE3.from_rotation_and_translation(
                tf.SO3(frame.wxyz), frame.position
            ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

            T_current_target = T_world_current.inverse() @ T_world_target

            for j in range(5):
                T_world_set = T_world_current @ tf.SE3.exp(
                    T_current_target.log() * j / 4.0
                )

                with client.atomic():
                    client.camera.wxyz = T_world_set.rotation().wxyz
                    client.camera.position = T_world_set.translation()

                time.sleep(1.0 / 15.0)
            client.camera.look_at = frame.position

        if not hasattr(self, "begin_call"):

            def begin_trans(client):
                assert client is not None
                T_world_current = tf.SE3.from_rotation_and_translation(
                    tf.SO3(client.camera.wxyz), client.camera.position
                )

                T_world_target = tf.SE3.from_rotation_and_translation(
                    tf.SO3(frame.wxyz), frame.position
                ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.5]))

                T_current_target = T_world_current.inverse() @ T_world_target

                for j in range(5):
                    T_world_set = T_world_current @ tf.SE3.exp(
                        T_current_target.log() * j / 4.0
                    )

                    with client.atomic():
                        client.camera.wxyz = T_world_set.rotation().wxyz
                        client.camera.position = T_world_set.translation()
                client.camera.look_at = frame.position

            self.begin_call = begin_trans

    def configure_optimizers(self):
        opt = OptimizationParams(
            parser=ArgumentParser(description="Training script parameters"),
            max_steps=self.edit_train_steps.value,
            lr_scaler=self.gs_lr_scaler.value,
            lr_final_scaler=self.gs_lr_end_scaler.value,
            color_lr_scaler=self.color_lr_scaler.value,
            opacity_lr_scaler=self.opacity_lr_scaler.value,
            scaling_lr_scaler=self.scaling_lr_scaler.value,
            rotation_lr_scaler=self.rotation_lr_scaler.value,
        )
        opt = OmegaConf.create(vars(opt))
        self.gaussian.spatial_lr_scale = self.cameras_extent
        self.gaussian.training_setup(opt)

    def render(
        self,
        cam,
        local=False,
        sam=False,
        train=False,
    ) -> Dict[str, Any]:
        self.gaussian.localize = local

        render_pkg = render(cam, self.gaussian, self.pipe, self.background_tensor)
        image, viewspace_point_tensor, _, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )
        if train:
            self.viewspace_point_tensor = viewspace_point_tensor
            self.radii = radii
            self.visibility_filter = self.radii > 0.0

        semantic_map = render(
            cam,
            self.gaussian,
            self.pipe,
            self.background_tensor,
            override_color=self.gaussian.mask.view(-1)[..., None].float().repeat(1, 3),
        )["render"]
        semantic_map = torch.norm(semantic_map, dim=0)
        semantic_map = semantic_map > 0.0  # 1, H, W
        semantic_map_viz = image.detach().clone()  # C, H, W
        semantic_map_viz = semantic_map_viz.permute(1, 2, 0)  # 3 512 512 to 512 512 3
        semantic_map_viz[semantic_map] = 0.50 * semantic_map_viz[
            semantic_map
        ] + 0.50 * torch.tensor([1.0, 0.0, 0.0], device="cuda")
        semantic_map_viz = semantic_map_viz.permute(2, 0, 1)  # 512 512 3 to 3 512 512

        render_pkg["sam_masks"] = []
        render_pkg["point2ds"] = []
        if sam:
            if hasattr(self, "points3d") and len(self.points3d) > 0:
                sam_output = self.sam_predict(image, cam)
                if sam_output is not None:
                    render_pkg["sam_masks"].append(sam_output[0])
                    render_pkg["point2ds"].append(sam_output[1])

        self.gaussian.localize = False  # reverse

        render_pkg["semantic"] = semantic_map_viz[None]
        render_pkg["masks"] = semantic_map[None]  # 1, 1, H, W

        image = image.permute(1, 2, 0)[None]  # C H W to 1 H W C
        render_pkg["comp_rgb"] = image  # 1 H W C

        depth = render_pkg["depth_3dgs"]
        depth = depth.permute(1, 2, 0)[None]
        render_pkg["depth"] = depth
        render_pkg["opacity"] = depth / (depth.max() + 1e-5)

        return {
            **render_pkg,
        }

    @torch.no_grad()
    def update_mask(self, text_prompt) -> None:
        print("\n" + "="*60)
        print("Starting Segmentation")
        print("="*60)

        # ========== SceneSplat 3D Segmentation (skip rendering entirely) ==========
        if self.use_scenesplat_checkbox.value:
            print(f"[SceneSplat] Direct 3D segmentation with prompt: '{text_prompt}'")

            ckpt_path = self.scenesplat_ckpt_path.value
            if self.scenesplat_segmentor is None or self.scenesplat_segmentor._ckpt_path != ckpt_path:
                self.seg_status.value = "正在加载模型，请稍等..."
                self.seg_status.visible = True
                while self.scenesplat_segmentor is None:
                    time.sleep(0.1)
                if self.scenesplat_segmentor._ckpt_path != ckpt_path:
                    self.scenesplat_segmentor = SceneSplatSegmentor(
                        checkpoint_path=ckpt_path, device="cuda:0"
                    )
                    self.scenesplat_segmentor._ckpt_path = ckpt_path

            # Interactive threshold adjustment
            self.seg_scale_end_button.visible = True
            self.mask_thres.visible = True
            self.seg_status.visible = True
            self.seg_status.value = "正在分割，请稍等..."
            self.show_semantic_mask.value = True

            selected_mask = self.scenesplat_segmentor.segment(
                self.gaussian, text_prompt, threshold=self.mask_thres.value
            )
            num_selected = selected_mask.sum().item()
            print(f"[SceneSplat] Selected {num_selected} Gaussians (threshold={self.mask_thres.value})")
            self.seg_status.value = f"分割完成，选中 {num_selected} 个 Gaussian"

            self.gaussian.set_mask(selected_mask)
            self.gaussian.apply_grad_mask(selected_mask)

            self.seg_scale = False
            while True:
                if self.seg_scale:
                    self.seg_status.value = "正在重新分割，请稍等..."
                    selected_mask = self.scenesplat_segmentor.segment(
                        self.gaussian, text_prompt, threshold=self.mask_thres.value
                    )
                    self.gaussian.set_mask(selected_mask)
                    self.gaussian.apply_grad_mask(selected_mask)
                    num_selected = selected_mask.sum().item()
                    print(f"[SceneSplat] Re-segmented: {num_selected} Gaussians (threshold={self.mask_thres.value})")
                    self.seg_status.value = f"分割完成，选中 {num_selected} 个 Gaussian"
                    self.seg_scale = False
                if self.seg_scale_end:
                    self.seg_scale_end = False
                    break
                time.sleep(0.01)

            self.seg_scale_end_button.visible = False
            self.mask_thres.visible = False
            self.seg_status.visible = False

            return [], selected_mask

        # ========== Original Two-Stage LangSAM Segmentation ==========
        print("Starting Two-Stage Adaptive Segmentation")

        # ========== Coordinate System Diagnosis ==========
        print("\n[Coordinate System Diagnosis]")
        xyz = self.gaussian.get_xyz.detach().cpu().numpy()
        x_min, x_max = xyz[:,0].min(), xyz[:,0].max()
        y_min, y_max = xyz[:,1].min(), xyz[:,1].max()
        z_min, z_max = xyz[:,2].min(), xyz[:,2].max()

        x_span = x_max - x_min
        y_span = y_max - y_min
        z_span = z_max - z_min

        print(f"Point cloud bounding box:")
        print(f"  X range: [{x_min:7.3f}, {x_max:7.3f}], span: {x_span:.3f}")
        print(f"  Y range: [{y_min:7.3f}, {y_max:7.3f}], span: {y_span:.3f}")
        print(f"  Z range: [{z_min:7.3f}, {z_max:7.3f}], span: {z_span:.3f}")

        # Determine vertical axis (usually the one with smallest span for indoor scenes)
        spans = {'X': x_span, 'Y': y_span, 'Z': z_span}
        vertical_axis = min(spans, key=spans.get)
        horizontal_axes = [ax for ax in ['X', 'Y', 'Z'] if ax != vertical_axis]

        print(f"  Likely vertical axis: {vertical_axis} (smallest span: {spans[vertical_axis]:.3f})")
        print(f"  Horizontal plane: {horizontal_axes[0]}-{horizontal_axes[1]}")
        print(f"  Scene center: [{(x_min+x_max)/2:.3f}, {(y_min+y_max)/2:.3f}, {(z_min+z_max)/2:.3f}]")
        print(f"  Total points: {len(xyz)}")

        # ========== Stage 1: Coarse Segmentation ==========
        print("\n[Stage 1] Coarse segmentation with indoor cameras...")

        # Generate indoor cameras based on bounding box
        bbox_min = np.array([x_min, y_min, z_min])
        bbox_max = np.array([x_max, y_max, z_max])

        indoor_cameras = generate_indoor_cameras(
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            vertical_axis=vertical_axis,
            grid_size=3,  # 3x3 grid = 9 positions
            image_height=512,
            image_width=512
        )

        print(f"[Stage 1] Generated {len(indoor_cameras)} indoor cameras (3x3 grid, 8 directions each)")

        coarse_masks = []
        coarse_weights = torch.zeros_like(self.gaussian._opacity)
        coarse_weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)

        # Use a subset of indoor cameras for coarse segmentation
        coarse_view_num = min(24, len(indoor_cameras))
        # Sample evenly from indoor cameras
        coarse_view_indices = [int(i * len(indoor_cameras) / coarse_view_num) for i in range(coarse_view_num)]

        print(f"[Stage 1] Using {coarse_view_num} cameras from {len(indoor_cameras)} total indoor cameras")

        # Create debug directory
        debug_dir = "/workspace/debug_segmentation"
        os.makedirs(debug_dir, exist_ok=True)
        print(f"[Stage 1] Debug images will be saved to: {debug_dir}")

        # Track valid detections
        valid_detection_count = 0

        for idx in tqdm(coarse_view_indices, desc="Coarse segmentation"):
            cur_cam = indoor_cameras[idx]
            this_frame = render(
                cur_cam, self.gaussian, self.pipe, self.background_tensor
            )["render"]

            this_frame = this_frame.moveaxis(0, -1)[None, ...]
            mask = self.text_segmentor(this_frame, text_prompt)[0].to(get_device())

            # Calculate mask statistics
            mask_sum = mask.sum().item()
            mask_area = int(mask_sum)
            total_pixels = mask.numel()
            coverage_ratio = mask_sum / total_pixels if total_pixels > 0 else 0

            # Check if this is a valid detection (non-zero mask)
            is_valid = mask_sum > 0
            if is_valid:
                valid_detection_count += 1

            # Print detection info
            status = "✓ DETECTED" if is_valid else "✗ None"
            print(f"  View {idx:3d}: {status} | Area: {mask_area:6d} pixels | Coverage: {coverage_ratio*100:.2f}%")

            # Save debug images
            # Save rendered image
            render_img = this_frame[0].moveaxis(-1, 0)  # H W C -> C H W
            torchvision.utils.save_image(
                render_img.clamp(0, 1),
                os.path.join(debug_dir, f"coarse_view{idx:03d}_render.png")
            )

            # Save mask (only if detected)
            if is_valid:
                # Create a colored overlay: red mask on grayscale image
                gray_img = render_img.mean(dim=0, keepdim=True).repeat(3, 1, 1)
                mask_overlay = gray_img.clone()
                mask_overlay[0] = torch.where(mask[0] > 0.5, torch.tensor(1.0), gray_img[0])  # Red channel
                mask_overlay[1] = torch.where(mask[0] > 0.5, torch.tensor(0.0), gray_img[1])  # Green channel
                mask_overlay[2] = torch.where(mask[0] > 0.5, torch.tensor(0.0), gray_img[2])  # Blue channel

                torchvision.utils.save_image(
                    mask_overlay,
                    os.path.join(debug_dir, f"coarse_view{idx:03d}_mask_overlay.png")
                )

                # Save pure mask
                torchvision.utils.save_image(
                    mask.float(),
                    os.path.join(debug_dir, f"coarse_view{idx:03d}_mask.png")
                )

            coarse_masks.append(mask)
            self.gaussian.apply_weights(cur_cam, coarse_weights, coarse_weights_cnt, mask)

        print(f"[Stage 1] Valid detections: {valid_detection_count}/{coarse_view_num}")
        print(f"[Stage 1] Debug images saved to: {debug_dir}")

        # Check if coarse segmentation failed
        if valid_detection_count < 5:
            print(f"[Stage 1] ERROR: Too few valid detections ({valid_detection_count}/{coarse_view_num})")
            print(f"[Stage 1] Coarse segmentation failed. Possible reasons:")
            print(f"  - Text prompt '{text_prompt}' doesn't match any object in the scene")
            print(f"  - Object is too small or occluded in most views")
            print(f"  - LangSAM detection thresholds are too high")
            print(f"[Stage 1] Aborting segmentation. Please try:")
            print(f"  - A more specific or different text prompt")
            print(f"  - Lowering box_threshold and text_threshold in LangSAM")

            # Return empty result
            self.seg_scale_end_button.visible = True
            self.mask_thres.visible = True
            self.show_semantic_mask.value = True

            empty_mask = torch.zeros_like(self.gaussian._opacity[:, 0], dtype=torch.bool)
            self.gaussian.set_mask(empty_mask)
            self.gaussian.apply_grad_mask(empty_mask)

            while True:
                if self.seg_scale_end:
                    self.seg_scale_end = False
                    break
                time.sleep(0.01)

            self.seg_scale_end_button.visible = False
            self.mask_thres.visible = False
            return coarse_masks, empty_mask

        coarse_weights /= coarse_weights_cnt + 1e-7

        # Get coarse mask with a higher threshold for better precision
        coarse_threshold = 0.6  # Require detection in 60% of views
        coarse_mask = coarse_weights > coarse_threshold
        coarse_mask = coarse_mask[:, 0]

        num_selected = coarse_mask.sum().item()
        print(f"[Stage 1] Coarse segmentation selected {num_selected} Gaussians (threshold={coarse_threshold})")

        if num_selected < 10:
            print(f"[Stage 1] WARNING: Very few Gaussians selected ({num_selected})")
            print(f"[Stage 1] Trying lower threshold (0.4)...")

            # Try a lower threshold
            coarse_mask = coarse_weights > 0.4
            coarse_mask = coarse_mask[:, 0]
            num_selected = coarse_mask.sum().item()
            print(f"[Stage 1] With threshold=0.4, selected {num_selected} Gaussians")

            if num_selected < 10:
                print("[Stage 1] ERROR: Still too few Gaussians. Using global views only.")
                weights = coarse_weights
                weights_cnt = coarse_weights_cnt
                masks = coarse_masks
            else:
                # Proceed with stage 2
                weights, weights_cnt, masks = self._run_fine_segmentation(
                    text_prompt, coarse_mask, coarse_weights, coarse_weights_cnt, coarse_masks
                )
        else:
            # Proceed with stage 2
            weights, weights_cnt, masks = self._run_fine_segmentation(
                text_prompt, coarse_mask, coarse_weights, coarse_weights_cnt, coarse_masks
            )

        print("\n" + "="*60)
        print("Two-Stage Segmentation Complete")
        print("="*60 + "\n")

        # Interactive threshold adjustment
        self.seg_scale_end_button.visible = True
        self.mask_thres.visible = True
        self.show_semantic_mask.value = True
        while True:
            if self.seg_scale:
                selected_mask = weights > self.mask_thres.value
                selected_mask = selected_mask[:, 0]
                self.gaussian.set_mask(selected_mask)
                self.gaussian.apply_grad_mask(selected_mask)

                self.seg_scale = False
            if self.seg_scale_end:
                self.seg_scale_end = False
                break
            time.sleep(0.01)

        self.seg_scale_end_button.visible = False
        self.mask_thres.visible = False
        return masks, selected_mask

    def _run_fine_segmentation(self, text_prompt, coarse_mask, coarse_weights, coarse_weights_cnt, coarse_masks):
        """Helper method to run fine segmentation stage"""
        # Calculate target object center and radius
        selected_points = self.gaussian.get_xyz[coarse_mask]
        target_center = selected_points.mean(dim=0).detach().cpu().numpy()

        # Use 75th percentile for radius to be more conservative
        dists = torch.norm(selected_points - selected_points.mean(dim=0), dim=1)
        target_radius = float(torch.quantile(dists, 0.75).cpu())

        # Get scene center
        if hasattr(self, 'scene_center') and self.scene_center is not None:
            scene_center = self.scene_center
        else:
            scene_center = self.gaussian.get_xyz.mean(dim=0).detach().cpu().numpy()

        print(f"[Stage 1] Target center: {target_center}")
        print(f"[Stage 1] Target radius: {target_radius:.3f}")
        print(f"[Stage 1] Scene center: {scene_center}")

        # ========== Stage 2: Fine Segmentation ==========
        print("\n[Stage 2] Fine segmentation with adaptive views...")

        adaptive_cameras = generate_adaptive_cameras(
            target_center=target_center,
            target_radius=target_radius,
            scene_center=scene_center,
            n_cameras=80,
            image_height=512,
            image_width=512,
            distance_factor=2.5
        )

        print(f"[Stage 2] Generated {len(adaptive_cameras)} adaptive cameras")

        # Fine segmentation with adaptive cameras
        fine_masks = []
        fine_weights = torch.zeros_like(self.gaussian._opacity)
        fine_weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)

        for idx, cur_cam in enumerate(tqdm(adaptive_cameras, desc="Fine segmentation")):
            this_frame = render(
                cur_cam, self.gaussian, self.pipe, self.background_tensor
            )["render"]

            this_frame = this_frame.moveaxis(0, -1)[None, ...]
            mask = self.text_segmentor(this_frame, text_prompt)[0].to(get_device())

            if self.use_sam:
                print("Using SAM")
                self.sam_features[idx] = self.sam_predictor._features

            fine_masks.append(mask)
            self.gaussian.apply_weights(cur_cam, fine_weights, fine_weights_cnt, mask)

        fine_weights /= fine_weights_cnt + 1e-7

        # Combine coarse and fine weights (weighted average)
        # Give more weight to fine segmentation
        weights = 0.3 * coarse_weights + 0.7 * fine_weights
        weights_cnt = coarse_weights_cnt + fine_weights_cnt
        masks = coarse_masks + fine_masks

        print(f"[Stage 2] Fine segmentation completed")
        print(f"[Stage 2] Total masks collected: {len(masks)}")

        return weights, weights_cnt, masks

    @property
    def camera(self):
        if len(list(self.server.get_clients().values())) == 0:
            return None
        
        # 无论是否有 colmap_dir，都需要初始化 aspect
        if not hasattr(self, 'aspect'):
            self.aspect = list(self.server.get_clients().values())[0].camera.aspect
        
        if self.render_cameras is None and self.colmap_dir is not None:
            self.render_cameras = CamScene(
                self.colmap_dir, h=-1, w=-1, aspect=self.aspect
            ).cameras
            self.begin_call(list(self.server.get_clients().values())[0])

        viser_cam = list(self.server.get_clients().values())[0].camera
        R = tf.SO3(viser_cam.wxyz).as_matrix()
        T = -R.T @ viser_cam.position
        if self.render_cameras is None:
            fovy = viser_cam.fov * self.FoV_slider.value
        else:
            fovy = self.render_cameras[0].FoVy * self.FoV_slider.value

        fovx = 2 * math.atan(math.tan(fovy / 2) * self.aspect)
        width = int(self.resolution_slider.value)
        height = int(width / self.aspect)
        return Simple_Camera(0, R, T, fovx, fovy, height, width, "", 0)

    def click_cb(self, pointer):
        if self.sam_enabled.value and self.add_sam_points.value:
            assert hasattr(pointer, "click_pos"), "please install our forked viser"
            click_pos = pointer.click_pos  # tuple (float, float)  W, H from 0 to 1
            click_pos = torch.tensor(click_pos)

            self.add_points3d(self.camera, click_pos)

            self.viwer_need_update = True
        elif self.draw_bbox.value:
            assert hasattr(pointer, "click_pos"), "please install our forked viser"
            click_pos = pointer.click_pos
            click_pos = torch.tensor(click_pos)
            cur_cam = self.camera
            if self.draw_flag:
                self.left_up.value = [
                    int(cur_cam.image_width * click_pos[0]),
                    int(cur_cam.image_height * click_pos[1]),
                ]
                self.draw_flag = False
            else:
                new_value = [
                    int(cur_cam.image_width * click_pos[0]),
                    int(cur_cam.image_height * click_pos[1]),
                ]
                if (self.left_up.value[0] < new_value[0]) and (
                    self.left_up.value[1] < new_value[1]
                ):
                    self.right_down.value = new_value
                    self.draw_flag = True
                else:
                    self.left_up.value = new_value

    def set_system(self, system):
        self.system = system

    def clear_points3d(self):
        self.points3d = []

    def add_points3d(self, camera, points2d, update_mask=False):
        depth = render(camera, self.gaussian, self.pipe, self.background_tensor)[
            "depth_3dgs"
        ]
        unprojected_points3d = unproject(camera, points2d, depth)
        self.points3d += unprojected_points3d.unbind(0)

        if update_mask:
            self.update_sam_mask_with_point_prompt(self.points3d)

    @torch.no_grad()
    def update_sam_mask_with_point_prompt(
        self, points3d=None, save_mask=False, save_name="point_prompt_mask"
    ):
        points3d = points3d if points3d is not None else self.points3d
        masks = []
        weights = torch.zeros_like(self.gaussian._opacity)
        weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)

        total_view_num = len(self.colmap_cameras)
        random.seed(0)  # make sure same views
        view_index = random.sample(
            range(0, total_view_num),
            min(total_view_num, self.seg_cam_num.value),
        )
        for idx in tqdm(view_index):
            cur_cam = self.colmap_cameras[idx]
            assert len(points3d) > 0
            points2ds = project(cur_cam, points3d)
            img = render(cur_cam, self.gaussian, self.pipe, self.background_tensor)[
                "render"
            ]

            self.sam_predictor.set_image(
                np.asarray(to_pil_image(img.cpu())),
            )
            self.sam_features[idx] = self.sam_predictor._features
            mask, _, _ = self.sam_predictor.predict(
                point_coords=points2ds.cpu().numpy(),
                point_labels=np.array([1] * points2ds.shape[0], dtype=np.int64),
                box=None,
                multimask_output=False,
            )
            mask = torch.from_numpy(mask).to(torch.bool).to(get_device())
            self.gaussian.apply_weights(
                cur_cam, weights, weights_cnt, mask.to(torch.float32)
            )
            masks.append(mask)

        weights /= weights_cnt + 1e-7

        self.seg_scale_end_button.visible = True
        self.mask_thres.visible = True
        self.show_semantic_mask.value = True
        while True:
            if self.seg_scale:
                selected_mask = weights > self.mask_thres.value
                selected_mask = selected_mask[:, 0]
                self.gaussian.set_mask(selected_mask)
                self.gaussian.apply_grad_mask(selected_mask)

                self.seg_scale = False
            if self.seg_scale_end:
                self.seg_scale_end = False
                break
            time.sleep(0.01)

        self.seg_scale_end_button.visible = False
        self.mask_thres.visible = False

        if save_mask:
            for id, mask in enumerate(masks):
                mask = mask.cpu().numpy()[0, 0]
                img = Image.fromarray(mask)
                os.makedirs("tmp", exist_ok=True)
                img.save(f"./tmp/{save_name}-{id}.jpg")

        return masks, selected_mask

    @torch.no_grad()
    def sam_predict(self, image, cam):
        img = np.asarray(to_pil_image(image.cpu()))
        self.sam_predictor.set_image(img)
        if len(self.points3d) == 0:
            return
        _points2ds = project(cam, self.points3d)
        _mask, _, _ = self.sam_predictor.predict(
            point_coords=_points2ds.cpu().numpy(),
            point_labels=np.array([1] * _points2ds.shape[0], dtype=np.int64),
            box=None,
            multimask_output=False,
        )
        _mask = torch.from_numpy(_mask).to(torch.bool).to(get_device())

        return _mask.squeeze(), _points2ds

    @torch.no_grad()
    def prepare_output_image(self, output):
        out_key = self.renderer_output.value
        out_img = output[out_key][0]  # H W C
        if out_key == "comp_rgb":
            if self.show_semantic_mask.value:
                out_img = output["semantic"][0].moveaxis(0, -1)
        elif out_key == "masks":
            out_img = output["masks"][0].to(torch.float32)[..., None].repeat(1, 1, 3)
        if out_img.dtype == torch.float32:
            out_img = out_img.clamp(0, 1)
            out_img = (out_img * 255).to(torch.uint8).cpu().to(torch.uint8)
            out_img = out_img.moveaxis(-1, 0)  # C H W

        if self.sam_enabled.value:
            if "sam_masks" in output and len(output["sam_masks"]) > 0:
                try:
                    out_img = torchvision.utils.draw_segmentation_masks(
                        out_img, output["sam_masks"][0]
                    )

                    out_img = torchvision.utils.draw_keypoints(
                        out_img,
                        output["point2ds"][0][None, ...],
                        colors="blue",
                        radius=5,
                    )
                except Exception as e:
                    print(e)

        if (
            self.draw_bbox.value
            and self.draw_flag
            and (self.left_up.value[0] < self.right_down.value[0])
            and (self.left_up.value[1] < self.right_down.value[1])
        ):
            out_img[
                :,
                self.left_up.value[1]: self.right_down.value[1],
                self.left_up.value[0]: self.right_down.value[0],
            ] = 0

        self.renderer_output.options = list(output.keys())
        return out_img.cpu().moveaxis(0, -1).numpy().astype(np.uint8)

    def render_loop(self):
        while True:
            self.update_viewer()
            time.sleep(1e-3)

    @torch.no_grad()
    def update_viewer(self):
        gs_camera = self.camera
        if gs_camera is None:
            return
        output = self.render(gs_camera, sam=self.sam_enabled.value)

        out = self.prepare_output_image(output)
        self.server.set_background_image(out, format="jpeg")

    def delete(self, edit_cameras, train_frames, train_frustums):
        if not self.ctn_inpaint:
            from diffusers import (
                StableDiffusionControlNetInpaintPipeline,
                ControlNetModel,
                DDIMScheduler,
            )

            controlnet = ControlNetModel.from_pretrained(
                "/workspace/.cache/huggingface/hub/models--lllyasviel--control_v11p_sd15_inpaint/snapshots/c96e03a807e64135568ba8aecb66b3a306ec73bd",
                torch_dtype=torch.float16, local_files_only=True
            )
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "/workspace/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
                controlnet=controlnet,
                torch_dtype=torch.float16,
                local_files_only=True,
            )
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            pipe.enable_model_cpu_offload()
            self.ctn_inpaint = pipe
            self.ctn_inpaint.set_progress_bar_config(disable=True)
            self.ctn_inpaint.safety_checker = None

        num_channels_latents = self.ctn_inpaint.vae.config.latent_channels
        shape = (
            1,
            num_channels_latents,
            edit_cameras[0].image_height // self.ctn_inpaint.vae_scale_factor,
            edit_cameras[0].image_height // self.ctn_inpaint.vae_scale_factor,
        )

        latents = torch.zeros(shape, dtype=torch.float16, device="cuda")

        dist_thres = (
            self.inpaint_scale.value * self.cameras_extent * self.gaussian.percent_dense
        )
        valid_remaining_idx = self.gaussian.get_near_gaussians_by_mask(
            self.gaussian.mask, dist_thres
        )
        self.gaussian.prune_with_mask(new_mask=valid_remaining_idx)

        inpaint_2D_mask, origin_frames = self.render_all_view_with_mask(
            edit_cameras, train_frames, train_frustums
        )

        self.guidance = DelGuidance(
            guidance=self.ctn_inpaint,
            latents=latents,
            gaussian=self.gaussian,
            text_prompt=self.edit_text.value,
            lambda_l1=self.lambda_l1.value,
            lambda_p=self.lambda_p.value,
            lambda_anchor_color=self.lambda_anchor_color.value,
            lambda_anchor_geo=self.lambda_anchor_geo.value,
            lambda_anchor_scale=self.lambda_anchor_scale.value,
            lambda_anchor_opacity=self.lambda_anchor_opacity.value,
            train_frames=train_frames,
            train_frustums=train_frustums,
            cams=edit_cameras,
            server=self.server,
        )

        view_index_stack = list(range(len(edit_cameras)))
        for step in tqdm(range(self.edit_train_steps.value)):
            if not view_index_stack:
                view_index_stack = list(range(len(edit_cameras)))
            view_index = random.choice(view_index_stack)
            view_index_stack.remove(view_index)

            rendering = self.render(edit_cameras[view_index], train=True)["comp_rgb"]

            loss = self.guidance(
                rendering,
                origin_frames[view_index],
                inpaint_2D_mask[view_index],
                view_index,
                step,
            )
            loss.backward()

            self.densify_and_prune(step)

            self.gaussian.optimizer.step()
            self.gaussian.optimizer.zero_grad(set_to_none=True)
            if self.stop_training:
                self.stop_training = False
                return

    def edit(self, edit_cameras, train_frames, train_frustums):
        # Debug: Print dimensions to diagnose potential mismatches
        print(f"[DEBUG] edit_cameras length: {len(edit_cameras)}")
        print(f"[DEBUG] train_frames length: {len(train_frames)}")
        print(f"[DEBUG] train_frustums length: {len(train_frustums)}")

        # Save rendered views from all edit cameras for local inspection
        import os
        from torchvision.utils import save_image
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_views")
        os.makedirs(save_dir, exist_ok=True)
        print(f"[Edit] Saving {len(edit_cameras)} camera views to {save_dir} ...")
        for idx, cam in enumerate(edit_cameras):
            with torch.no_grad():
                out = self.render(cam)["comp_rgb"]  # [1, H, W, 3]
                img = out.squeeze(0).permute(2, 0, 1).clamp(0, 1)  # [3, H, W]
                save_image(img, os.path.join(save_dir, f"view_{idx:03d}.png"))
        print(f"[Edit] All {len(edit_cameras)} views saved to {save_dir}")

        if self.guidance_type.value == "InstructPix2Pix":
            if not self.ip2p:
                from threestudio.models.guidance.instructpix2pix_guidance import (
                    InstructPix2PixGuidance,
                )

                self.ip2p = InstructPix2PixGuidance(
                    OmegaConf.create({"min_step_percent": 0.02, "max_step_percent": 0.98})
                )
            cur_2D_guidance = self.ip2p
            print("using InstructPix2Pix!")
        elif self.guidance_type.value == "ControlNet-Pix2Pix":
            if not self.ctn_ip2p:
                from threestudio.models.guidance.controlnet_guidance import (
                    ControlNetGuidance,
                )

                self.ctn_ip2p = ControlNetGuidance(
                    OmegaConf.create({"min_step_percent": 0.05,
                                      "max_step_percent": 0.8,
                                      "control_type": "p2p"})
                )
            cur_2D_guidance = self.ctn_ip2p
            print("using ControlNet-InstructPix2Pix!")

        origin_frames = self.render_cameras_list(edit_cameras)
        self.guidance = EditGuidance(
            guidance=cur_2D_guidance,
            gaussian=self.gaussian,
            origin_frames=origin_frames,
            text_prompt=self.edit_text.value,
            per_editing_step=self.per_editing_step.value,
            edit_begin_step=self.edit_begin_step.value,
            edit_until_step=self.edit_until_step.value,
            lambda_l1=self.lambda_l1.value,
            lambda_p=self.lambda_p.value,
            lambda_anchor_color=self.lambda_anchor_color.value,
            lambda_anchor_geo=self.lambda_anchor_geo.value,
            lambda_anchor_scale=self.lambda_anchor_scale.value,
            lambda_anchor_opacity=self.lambda_anchor_opacity.value,
            train_frames=train_frames,
            train_frustums=train_frustums,
            cams=edit_cameras,
            server=self.server,
            lambda_clip_consistency=self.lambda_consistency.value if self.use_multiview_consistency.value else 0.0,
            pre_generate=self.pre_generate_edits.value,
        )
        view_index_stack = list(range(len(edit_cameras)))
        for step in tqdm(range(self.edit_train_steps.value)):
            if not view_index_stack:
                view_index_stack = list(range(len(edit_cameras)))
            view_index = random.choice(view_index_stack)
            view_index_stack.remove(view_index)

            # Boundary check to prevent index out of bounds
            if view_index >= len(train_frames):
                print(f"Warning: view_index {view_index} >= {len(train_frames)}, skipping")
                continue

            rendering = self.render(edit_cameras[view_index], train=True)["comp_rgb"]

            loss = self.guidance(rendering, view_index, step)
            loss.backward()

            self.densify_and_prune(step)

            self.gaussian.optimizer.step()
            self.gaussian.optimizer.zero_grad(set_to_none=True)
            if self.stop_training:
                self.stop_training = False
                return

    @torch.no_grad()
    def add(self, cam):
        self.draw_bbox.value = False
        self.inpaint_seed.visible = True
        self.inpaint_end.visible = True
        self.refine_text.visible = True
        self.draw_bbox.visible = False
        self.left_up.visible = False
        self.right_down.visible = False

        if not self.ctn_inpaint:
            from diffusers import (
                StableDiffusionControlNetInpaintPipeline,
                ControlNetModel,
                DDIMScheduler,
            )

            controlnet = ControlNetModel.from_pretrained(
                "/workspace/.cache/huggingface/hub/models--lllyasviel--control_v11p_sd15_inpaint/snapshots/c96e03a807e64135568ba8aecb66b3a306ec73bd",
                torch_dtype=torch.float16, local_files_only=True
            )
            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                "/workspace/.cache/huggingface/hub/models--runwayml--stable-diffusion-v1-5/snapshots/451f4fe16113bff5a5d2269ed5ad43b0592e9a14",
                controlnet=controlnet,
                torch_dtype=torch.float16,
                local_files_only=True,
            )
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            pipe.enable_model_cpu_offload()
            self.ctn_inpaint = pipe
            self.ctn_inpaint.set_progress_bar_config(disable=True)
            self.ctn_inpaint.safety_checker = None

        with torch.no_grad():
            render_pkg = render(cam, self.gaussian, self.pipe, self.background_tensor)

        image_in = to_pil_image(torch.clip(render_pkg["render"], 0.0, 1.0))
        origin_size = image_in.size  # W, H

        frustum = None
        frame = None
        while True:
            if self.inpaint_again:
                if frustum is not None:
                    frustum.remove()
                    frame.remove()
                mask_in = torch.zeros(
                    (origin_size[1], origin_size[0]),
                    dtype=torch.float32,
                    device=get_device(),
                )  # H, W
                mask_in[
                    self.left_up.value[1]: self.right_down.value[1],
                    self.left_up.value[0]: self.right_down.value[0],
                ] = 1.0

                image_in_pil = to_pil_image(
                    ui_utils.resize_image_ctn(np.asarray(image_in), 512)
                )
                mask_in = to_pil_image(mask_in)
                mask_in_pil = to_pil_image(
                    ui_utils.resize_image_ctn(np.asarray(mask_in)[..., None], 512)
                )

                image = np.array(image_in_pil.convert("RGB")).astype(np.float32) / 255.0
                image_mask = (
                    np.array(mask_in_pil.convert("L")).astype(np.float32) / 255.0
                )

                image[image_mask > 0.5] = -1.0  # set as masked pixel
                image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
                control_image = torch.from_numpy(image).to("cuda")
                generator = torch.Generator(device="cuda").manual_seed(
                    self.inpaint_seed.value
                )
                out = self.ctn_inpaint(
                    self.edit_text.value + ", high quality, extremely detailed",
                    num_inference_steps=25,
                    generator=generator,
                    eta=1.0,
                    image=image_in_pil,
                    mask_image=mask_in_pil,
                    control_image=control_image,
                ).images[0]
                out = cv2.resize(
                    np.asarray(out),
                    origin_size,
                    interpolation=cv2.INTER_LANCZOS4
                    if out.width / origin_size[0] > 1
                    else cv2.INTER_AREA,
                )
                out = to_pil_image(out)
                frame, frustum = ui_utils.new_frustum_from_cam(
                    list(self.server.get_clients().values())[0].camera,
                    self.server,
                    np.asarray(out),
                )
                self.inpaint_again = False
            else:
                if self.stop_training:
                    self.stop_training = False
                    return
                time.sleep(0.1)
            if self.inpaint_end_flag:
                self.inpaint_end_flag = False
                break
        self.inpaint_seed.visible = False
        self.inpaint_end.visible = False
        self.edit_text.visible = False

        removed_bg = rembg.remove(out)
        inpainted_image = to_tensor(out).to("cuda")
        frustum.remove()
        frame.remove()
        frame, frustum = ui_utils.new_frustum_from_cam(
            list(self.server.get_clients().values())[0].camera,
            self.server,
            np.asarray(removed_bg),
        )

        cache_dir = Path("tmp_add").absolute().as_posix()
        os.makedirs(cache_dir, exist_ok=True)
        mv_image_dir = os.path.join(cache_dir, "multiview_pred_images")
        os.makedirs(mv_image_dir, exist_ok=True)
        inpaint_path = os.path.join(cache_dir, "inpainted.png")
        removed_bg_path = os.path.join(cache_dir, "removed_bg.png")
        mesh_path = os.path.join(cache_dir, "inpaint_mesh.obj")
        gs_path = os.path.join(cache_dir, "inpaint_gs.obj")
        out.save(inpaint_path)
        removed_bg.save(removed_bg_path)

        p1 = subprocess.Popen(
            f"{sys.prefix}/bin/accelerate launch --config_file 1gpu.yaml test_mvdiffusion_seq.py "
            f"--save_dir {mv_image_dir} --config configs/mvdiffusion-joint-ortho-6views.yaml"
            f" validation_dataset.root_dir={cache_dir} validation_dataset.filepaths=[removed_bg.png]".split(" "),
            cwd="threestudio/utils/wonder3D",
        )
        p1.wait()

        cmd = (
            f"{sys.prefix}/bin/python launch.py --config configs/neuralangelo-ortho-wmask.yaml "
            f"--save_dir {cache_dir} --gpu 0 --train "
            f"dataset.root_dir={os.path.dirname(mv_image_dir)} "
            f"dataset.scene={os.path.basename(mv_image_dir)}"
        ).split(" ")
        p2 = subprocess.Popen(cmd, cwd="threestudio/utils/wonder3D/instant-nsr-pl")
        p2.wait()

        p3 = subprocess.Popen(
            [
                f"{sys.prefix}/bin/python",
                "train_from_mesh.py",
                "--mesh", mesh_path,
                "--save_path", gs_path,
                "--prompt", self.refine_text.value,
            ]
        )
        p3.wait()

        frustum.remove()
        frame.remove()

        object_mask = np.array(removed_bg)
        object_mask = object_mask[:, :, 3] > 0
        object_mask = torch.from_numpy(object_mask)
        bbox = masks_to_boxes(object_mask[None])[0].to("cuda")

        depth_estimator = DPT(get_device(), mode="depth")

        estimated_depth = depth_estimator(
            inpainted_image.moveaxis(0, -1)[None, ...]
        ).squeeze()
        object_center = (bbox[:2] + bbox[2:]) / 2

        fx = fov2focal(cam.FoVx, cam.image_width)
        fy = fov2focal(cam.FoVy, cam.image_height)

        object_center = (
            object_center
            - torch.tensor([cam.image_width, cam.image_height]).to("cuda") / 2
        ) / torch.tensor([fx, fy]).to("cuda")

        rendered_depth = render_pkg["depth_3dgs"][..., ~object_mask]

        inpainted_depth = estimated_depth[~object_mask]
        object_depth = estimated_depth[..., object_mask]

        min_object_depth = torch.quantile(object_depth, 0.05)
        max_object_depth = torch.quantile(object_depth, 0.95)
        obj_depth_scale = (max_object_depth - min_object_depth) * 1

        min_valid_depth_mask = (min_object_depth - obj_depth_scale) < inpainted_depth
        max_valid_depth_mask = inpainted_depth < (max_object_depth + obj_depth_scale)
        valid_depth_mask = torch.logical_and(min_valid_depth_mask, max_valid_depth_mask)
        valid_percent = valid_depth_mask.sum() / min_valid_depth_mask.shape[0]
        print("depth valid percent: ", valid_percent)

        rendered_depth = rendered_depth[0, valid_depth_mask]
        inpainted_depth = inpainted_depth[valid_depth_mask.squeeze()]

        y = rendered_depth
        x = inpainted_depth
        a = (torch.sum(x * y) - torch.sum(x) * torch.sum(y)) / (
            torch.sum(x ** 2) - torch.sum(x) ** 2
        )
        b = torch.sum(y) - a * torch.sum(x)

        z_in_cam = object_depth.min() * a + b

        self.depth_scaler.visible = True
        self.depth_end.visible = True
        self.refine_text.visible = True

        new_object_gaussian = None
        while True:
            if self.scale_depth:
                if new_object_gaussian is not None:
                    self.gaussian.prune_with_mask()
                scaled_z_in_cam = z_in_cam * self.depth_scaler.value
                x_in_cam, y_in_cam = (object_center.cuda()) * scaled_z_in_cam
                T_in_cam = torch.stack([x_in_cam, y_in_cam, scaled_z_in_cam], dim=-1)

                bbox = bbox.cuda()
                real_scale = (
                    (bbox[2:] - bbox[:2])
                    / torch.tensor([fx, fy], device="cuda")
                    * scaled_z_in_cam
                )

                new_object_gaussian = VanillaGaussianModel(self.gaussian.max_sh_degree)
                new_object_gaussian.load_ply(gs_path)
                new_object_gaussian._opacity.data = (
                    torch.ones_like(new_object_gaussian._opacity.data) * 99.99
                )

                new_object_gaussian._xyz.data -= new_object_gaussian._xyz.data.mean(
                    dim=0, keepdim=True
                )
                rotate_gaussians(new_object_gaussian, default_model_mtx.T)

                object_scale = (
                    new_object_gaussian._xyz.data.max(dim=0)[0]
                    - new_object_gaussian._xyz.data.min(dim=0)[0]
                )[:2]

                relative_scale = (real_scale / object_scale).mean()
                print(relative_scale)

                scale_gaussians(new_object_gaussian, relative_scale)

                new_object_gaussian._xyz.data += T_in_cam

                R = torch.from_numpy(cam.R).float().cuda()
                T = -R @ torch.from_numpy(cam.T).float().cuda()

                rotate_gaussians(new_object_gaussian, R)
                translate_gaussians(new_object_gaussian, T)

                self.gaussian.concat_gaussians(new_object_gaussian)
                self.scale_depth = False
            else:
                if self.stop_training:
                    self.stop_training = False
                    return
                time.sleep(0.01)
            if self.depth_end_flag:
                self.depth_end_flag = False
                break
        self.depth_scaler.visible = False
        self.depth_end.visible = False

    def densify_and_prune(self, step):
        if step <= self.densify_until_step.value:
            n_gaussians = self.gaussian.get_xyz.shape[0]
            # Guard against race condition: viewer render() may update
            # self.visibility_filter / self.radii concurrently with a
            # different Gaussian count (before/after densify).
            if (self.visibility_filter.shape[0] != n_gaussians
                    or self.radii.shape[0] != n_gaussians
                    or self.viewspace_point_tensor.shape[0] != n_gaussians):
                return  # sizes mismatch — skip this step safely

            self.gaussian.max_radii2D[self.visibility_filter] = torch.max(
                self.gaussian.max_radii2D[self.visibility_filter],
                self.radii[self.visibility_filter],
            )
            self.gaussian.add_densification_stats(
                self.viewspace_point_tensor.grad, self.visibility_filter
            )

            if step > 0 and step % self.densification_interval.value == 0:
                self.gaussian.densify_and_prune(
                    max_grad=1e-7,
                    max_densify_percent=self.max_densify_percent.value,
                    min_opacity=self.min_opacity.value,
                    extent=self.cameras_extent,
                    max_screen_size=5,
                )
                # Reset Adam momentum for non-masked Gaussians so their
                # accumulated momentum does not keep drifting them after grad masking.
                mask = self.gaussian.mask  # [N] bool
                if mask is not None:
                    for group in self.gaussian.optimizer.param_groups:
                        for param in group["params"]:
                            if param.shape[0] != mask.shape[0]:
                                continue
                            state = self.gaussian.optimizer.state.get(param)
                            if state is None:
                                continue
                            non_mask = ~mask
                            if "exp_avg" in state:
                                state["exp_avg"][non_mask] = 0.0
                            if "exp_avg_sq" in state:
                                state["exp_avg_sq"][non_mask] = 0.0

    @torch.no_grad()
    def render_cameras_list(self, edit_cameras):
        origin_frames = []
        for cam in edit_cameras:
            out = self.render(cam)["comp_rgb"]
            origin_frames.append(out)

        return origin_frames

    @torch.no_grad()
    def render_all_view_with_mask(self, edit_cameras, train_frames, train_frustums):
        inpaint_2D_mask = []
        origin_frames = []

        for idx, cam in enumerate(edit_cameras):
            res = self.render(cam)
            rgb, mask = res["comp_rgb"], res["masks"]
            mask = dilate_mask(mask.to(torch.float32), self.mask_dilate.value)
            if self.fix_holes.value:
                mask = fill_closed_areas(mask)
            inpaint_2D_mask.append(mask)
            origin_frames.append(rgb)
            train_frustums[idx].remove()
            mask_view = torch.stack([mask] * 3, dim=3)  # 1 H W C
            train_frustums[idx] = ui_utils.new_frustums(
                idx, train_frames[idx], cam, mask_view, True, self.server
            )
        return inpaint_2D_mask, origin_frames

    def add_theme(self):
        buttons = (
            TitlebarButton(
                text="Getting Started",
                icon=None,
                href="https://github.com/buaacyw/GaussianEditor/blob/master/docs/webui.md",
            ),
            TitlebarButton(
                text="Github",
                icon="GitHub",
                href="https://github.com/buaacyw/GaussianEditor",
            ),
            TitlebarButton(
                text="Hongjun Chen",
                icon=None,
                href="https://openreview.net/profile?id=%7EHongjun_Chen3",
            ),
        )
        image = TitlebarImage(
            image_url_light="https://s41.ax1x.com/2026/03/13/peE9gIK.png",
            image_alt="Logo",
            href="https://openreview.net/profile?id=%7EHongjun_Chen3",
        )
        titlebar_theme = TitlebarConfig(buttons=buttons, image=image)
        brand_color = self.server.add_gui_rgb("Brand color", (7, 0, 8), visible=False)

        self.server.configure_theme(
            titlebar_content=titlebar_theme,
            show_logo=True,
            brand_color=brand_color.value,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gs_source", type=str, required=True)  # gs ply or obj file?
    parser.add_argument("--colmap_dir", type=str, required=False, default=None)  # now optional

    args = parser.parse_args()
    webui = WebUI(args)
    webui.render_loop()
