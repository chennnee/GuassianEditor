import argparse
import json
import sys
import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from torchvision.transforms import ToPILImage, ToTensor
import cv2

from lang_sam import LangSAM

# Add SceneSplat to path
SCENESPLAT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '..', 'SceneSplat')
if os.path.exists(SCENESPLAT_ROOT):
    sys.path.insert(0, SCENESPLAT_ROOT)


class LangSAMTextSegmentor(torch.nn.Module):
    def __init__(
        self,
        sam_type="sam2.1_hiera_small",
        box_threshold=0.20,  # 降低阈值，增加检测灵敏度
        text_threshold=0.20,  # 降低阈值
        min_mask_area=50,    # 降低最小面积要求
        max_mask_area_ratio=0.95,  # 放宽最大面积限制
        use_morphology=False,  # 暂时关闭形态学处理，避免破坏细节
        morphology_kernel_size=3,  # 减小核大小
        debug=False,  # 是否打印调试信息
    ):
        super().__init__()
        self.model = LangSAM(sam_type)
        self.to_pil_image = ToPILImage(mode="RGB")
        self.to_tensor = ToTensor()

        # 阈值参数
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.min_mask_area = min_mask_area
        self.max_mask_area_ratio = max_mask_area_ratio

        # 后处理参数
        self.use_morphology = use_morphology
        self.morphology_kernel_size = morphology_kernel_size
        self.debug = debug

    def filter_masks(self, masks, scores, image_size):
        """
        根据面积和置信度过滤 masks

        Args:
            masks: numpy array of masks
            scores: confidence scores
            image_size: (H, W)

        Returns:
            filtered masks and scores
        """
        # 确保 masks 和 scores 是可迭代的数组
        masks = np.atleast_1d(masks)
        scores = np.atleast_1d(scores)

        # 如果 masks 是 2D (单个 mask)，转换为 3D
        if masks.ndim == 2:
            masks = masks[np.newaxis, ...]

        if len(masks) == 0:
            return masks, scores

        H, W = image_size
        total_pixels = H * W

        if self.debug:
            print(f"[Filter] Input: {len(masks)} masks, image size: {H}x{W}")

        filtered_masks = []
        filtered_scores = []

        for i, (mask, score) in enumerate(zip(masks, scores)):
            mask_area = mask.sum()
            area_ratio = mask_area / total_pixels

            if self.debug:
                print(f"  Mask {i}: score={score:.3f}, area={int(mask_area)}, ratio={area_ratio:.3f}")

            # 过滤太小的 mask
            if mask_area < self.min_mask_area:
                if self.debug:
                    print(f"    -> Filtered: too small (< {self.min_mask_area})")
                continue

            # 过滤太大的 mask（可能是背景）
            if mask_area > total_pixels * self.max_mask_area_ratio:
                if self.debug:
                    print(f"    -> Filtered: too large (> {self.max_mask_area_ratio:.2f})")
                continue

            if self.debug:
                print(f"    -> Kept")
            filtered_masks.append(mask)
            filtered_scores.append(score)

        if self.debug:
            print(f"[Filter] Output: {len(filtered_masks)} masks")

        if len(filtered_masks) == 0:
            return np.array([]), np.array([])

        return np.array(filtered_masks), np.array(filtered_scores)

    def morphology_postprocess(self, mask):
        """
        形态学后处理，去除噪声和填充空洞

        Args:
            mask: binary mask (H, W)

        Returns:
            processed mask
        """
        if not self.use_morphology:
            return mask

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.morphology_kernel_size, self.morphology_kernel_size)
        )

        # 转换为 uint8
        mask_uint8 = (mask * 255).astype(np.uint8)

        # 闭运算：先膨胀后腐蚀，填充小孔
        mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

        # 开运算：先腐蚀后膨胀，去除小噪声
        mask_opened = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

        return (mask_opened > 127).astype(np.float32)

    def select_best_mask(self, masks, scores):
        """
        从多个 masks 中选择最佳的一个

        策略：
        1. 优先选择置信度最高的
        2. 如果置信度相近，选择面积适中的

        Args:
            masks: numpy array of masks
            scores: confidence scores

        Returns:
            best mask index
        """
        if len(masks) == 0:
            return None

        if len(masks) == 1:
            if self.debug:
                print(f"[Select] Only 1 mask, selecting index 0")
            return 0

        # 计算每个 mask 的综合得分
        mask_areas = np.array([mask.sum() for mask in masks])

        # 归一化面积（避免极端值）
        area_mean = mask_areas.mean()
        area_std = mask_areas.std() + 1e-6
        area_scores = 1.0 - np.abs(mask_areas - area_mean) / (3 * area_std)
        area_scores = np.clip(area_scores, 0, 1)

        # 综合得分：置信度 * 0.8 + 面积得分 * 0.2 (更重视置信度)
        combined_scores = scores * 0.8 + area_scores * 0.2

        best_idx = np.argmax(combined_scores)

        if self.debug:
            print(f"[Select] Evaluating {len(masks)} masks:")
            for i in range(len(masks)):
                print(f"  Mask {i}: score={scores[i]:.3f}, area={int(mask_areas[i])}, "
                      f"area_score={area_scores[i]:.3f}, combined={combined_scores[i]:.3f}")
            print(f"  -> Selected mask {best_idx}")

        return best_idx

    def forward(self, images, prompt: str):
        images = rearrange(images, "b h w c -> b c h w")
        masks = []

        for idx, image in enumerate(images):
            image_pil = self.to_pil_image(image.clamp(0.0, 1.0))
            H, W = image_pil.size[1], image_pil.size[0]

            if self.debug:
                print(f"\n[Forward] Processing image {idx}, size: {W}x{H}, prompt: '{prompt}'")

            # 使用自定义阈值进行预测
            results = self.model.predict(
                [image_pil],
                [prompt],
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold
            )
            result = results[0]

            if self.debug:
                print(f"[Forward] LangSAM returned {len(result['masks'])} masks")

            if len(result["masks"]) > 0:
                # 获取 scores
                raw_scores = result.get("mask_scores", result.get("scores", np.ones(len(result["masks"]))))

                if self.debug:
                    print(f"[Forward] Raw scores: {raw_scores}")

                # 过滤 masks
                filtered_masks, filtered_scores = self.filter_masks(
                    result["masks"],
                    raw_scores,
                    (H, W)
                )

                if len(filtered_masks) > 0:
                    # 选择最佳 mask
                    best_idx = self.select_best_mask(filtered_masks, filtered_scores)

                    if best_idx is not None:
                        mask = filtered_masks[best_idx]

                        # 形态学后处理
                        mask = self.morphology_postprocess(mask)

                        mask = torch.from_numpy(mask[None].astype(np.float32))
                        masks.append(mask)

                        if self.debug:
                            print(f"[Forward] ✓ Successfully generated mask")
                    else:
                        if self.debug:
                            print(f"[Forward] ✗ No valid mask after selection")
                        print(f"No valid mask for '{prompt}' after filtering")
                        masks.append(torch.zeros_like(images[0, 0:1]))
                else:
                    if self.debug:
                        print(f"[Forward] ✗ All masks filtered out")
                    print(f"All masks filtered out for '{prompt}'")
                    masks.append(torch.zeros_like(images[0, 0:1]))
            else:
                if self.debug:
                    print(f"[Forward] ✗ No masks detected by LangSAM")
                print(f"None '{prompt}' Detected")
                masks.append(torch.zeros_like(images[0, 0:1]))

        return torch.stack(masks, dim=0)


class SceneSplatSegmentor(torch.nn.Module):
    """
    3D Gaussian segmentation using SceneSplat PT-v3 features + SigLIP2 text embeddings.
    Directly operates on Gaussian parameters, no rendering or cameras needed.
    """

    def __init__(self, checkpoint_path, config_path=None, device="cuda"):
        super().__init__()
        self.device = device

        # Save default dtype before loading models (transformers may change it)
        original_dtype = torch.get_default_dtype()

        # Load SceneSplat inference pipeline
        if config_path is None:
            config_path = os.path.join(
                SCENESPLAT_ROOT, "configs", "inference",
                "lang-pretrain-pt-v3m1-3dgs.py"
            )

        from pointcept.inference.lang_pretrainer import LangPretrainerInference
        self.inferencer = LangPretrainerInference(
            cfg=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
        )

        # Load SigLIP2 text encoder
        from transformers import AutoModel, AutoTokenizer
        siglip_name = "google/siglip2-base-patch16-512"
        self.text_model = AutoModel.from_pretrained(siglip_name).eval().to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(siglip_name)

        # Restore default dtype (prevents bfloat16 leaking into rasterizer)
        torch.set_default_dtype(original_dtype)

        # Cache for text embeddings
        self._text_cache = {}
        # Cache for backbone features (avoid re-running inference when only threshold changes)
        self._cached_similarity = None
        self._cached_num_points = None
        self._cached_prompt = None

    def _gaussian_to_numpy(self, gaussian_model):
        """Convert GaussianEditor's GaussianModel to SceneSplat numpy dict."""
        C0 = 0.28209479177387814

        xyz = gaussian_model._xyz.detach().cpu().numpy().astype(np.float32)

        features_dc = gaussian_model._features_dc.detach().cpu().numpy()
        if features_dc.ndim == 3:
            features_dc = features_dc.squeeze()
        color = np.clip(features_dc * C0 + 0.5, 0, 1) * 255
        color = color.astype(np.uint8)

        opacity_raw = gaussian_model._opacity.detach().cpu().numpy().astype(np.float32)
        opacity = 1.0 / (1.0 + np.exp(-opacity_raw))
        if opacity.ndim == 2:
            opacity = opacity.squeeze(-1)

        scale = np.exp(gaussian_model._scaling.detach().cpu().numpy().astype(np.float32))

        quat = gaussian_model._rotation.detach().cpu().numpy().astype(np.float32)
        quat_norm = np.linalg.norm(quat, axis=1, keepdims=True) + 1e-9
        quat = quat / quat_norm
        signs = np.sign(quat[:, 0])
        quat = quat * signs[:, None]

        return {
            "coord": xyz,
            "color": color,
            "opacity": opacity,
            "scale": scale,
            "quat": quat,
        }

    @torch.no_grad()
    def _encode_text(self, text_prompt):
        """Encode text prompt to SigLIP2 embedding [1, 768], with caching."""
        if text_prompt in self._text_cache:
            return self._text_cache[text_prompt]

        prompt = f"this is a {text_prompt}"
        inputs = self.tokenizer(
            [prompt], padding="max_length", max_length=64, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        text_embed = self.text_model.get_text_features(**inputs)
        text_embed = F.normalize(text_embed, dim=-1)  # [1, 768]

        self._text_cache[text_prompt] = text_embed
        return text_embed

    @torch.no_grad()
    def segment(self, gaussian_model, text_prompt, threshold=0.5):
        """
        Segment Gaussians by text prompt.

        Args:
            gaussian_model: GaussianModel with _xyz, _features_dc, _opacity, _scaling, _rotation
            text_prompt: text description of target object
            threshold: similarity threshold for mask

        Returns:
            [N] boolean mask on the same device as gaussian_model._xyz
        """
        num_points = gaussian_model._xyz.shape[0]

        # 1. 只有 Gaussian 数量变化或 prompt 变化时才重新推理，否则复用缓存
        need_rerun = (
            not hasattr(self, '_cached_similarity') or
            self._cached_similarity is None or
            self._cached_num_points != num_points or
            self._cached_prompt != text_prompt
        )

        if need_rerun:
            data_dict = self._gaussian_to_numpy(gaussian_model)
            self.inferencer.chunk_size = min(100000, num_points)
            torch.cuda.empty_cache()
            outputs = self.inferencer(data_dict, save=False)
            torch.cuda.empty_cache()
            features = torch.from_numpy(outputs["backbone_features"]).float().to(self.device)

            text_embed = self._encode_text(text_prompt)
            similarity = (features @ text_embed.T).squeeze(-1)

            # 缓存结果
            self._cached_similarity = similarity.cpu()
            self._cached_num_points = num_points
            self._cached_prompt = text_prompt
        else:
            similarity = self._cached_similarity.to(self.device)

        # 2. Z-score 阈值（每次都重新算，很快）
        z_score = 0.5 + (threshold - 0.2) / (0.99 - 0.2) * (4.0 - 0.5)
        mean = similarity.mean()
        std = similarity.std()
        cutoff = mean + z_score * std
        mask = similarity > cutoff

        return mask.to(gaussian_model._xyz.device)


if __name__ == "__main__":
    model = LangSAMTextSegmentor()
    image = Image.open("load/lego_bulldozer.jpg")
    prompt = "a lego bulldozer"
    image = ToTensor()(image).unsqueeze(0)
    mask = model(image, prompt)
    print(mask.shape)
