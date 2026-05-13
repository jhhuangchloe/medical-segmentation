#!/usr/bin/env python3
"""
Improved Liver Segmentation using SAM2 with better prompting strategy
Uses bounding box prompts instead of single points for more reliable segmentation
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm

# Add sam2 to path
sys.path.append('./sam2')

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


class ImprovedLiverSegmentorSAM2:
    """Improved liver segmentation using SAM2 with box prompts"""

    def __init__(self, model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
                 checkpoint="./sam2/checkpoints/sam2.1_hiera_large.pt",
                 device=None):
        """Initialize SAM2 liver segmentor"""
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Enable optimizations for CUDA
        if self.device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        # Build SAM2 model
        print("Loading SAM2 model...")
        print(f"Config: {model_cfg}")
        print(f"Checkpoint: {checkpoint}")

        if not os.path.exists(checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        sam2_model = build_sam2(model_cfg, checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(sam2_model)
        print("SAM2 model loaded successfully!")

    @staticmethod
    def read_nii(filepath):
        """Read .nii file and return pixel array"""
        ct_scan = nib.load(filepath)
        array = ct_scan.get_fdata()
        array = np.rot90(np.array(array))
        return array

    @staticmethod
    def normalize_ct_scan(image, window_center=30, window_width=150):
        """Normalize CT scan for liver viewing"""
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        windowed = np.clip(image, img_min, img_max)
        normalized = ((windowed - windowed.min()) /
                     (windowed.max() - windowed.min() + 1e-8) * 255).astype(np.uint8)
        return normalized

    @staticmethod
    def prepare_image_for_sam2(ct_slice, window_center=30, window_width=150):
        """Prepare CT slice for SAM2 input"""
        normalized = ImprovedLiverSegmentorSAM2.normalize_ct_scan(ct_slice, window_center, window_width)
        rgb_image = np.stack([normalized, normalized, normalized], axis=-1)
        return rgb_image

    @staticmethod
    def get_liver_bounding_box(mask, margin=20):
        """Get bounding box of the liver from ground truth mask"""
        liver_mask = (mask == 1).astype(np.uint8)
        if liver_mask.sum() == 0:
            return None

        y_coords, x_coords = np.where(liver_mask > 0)
        x_min = max(0, x_coords.min() - margin)
        x_max = min(mask.shape[1], x_coords.max() + margin)
        y_min = max(0, y_coords.min() - margin)
        y_max = min(mask.shape[0], y_coords.max() + margin)

        return np.array([x_min, y_min, x_max, y_max])

    @staticmethod
    def get_liver_multiple_points(mask, num_points=5):
        """Get multiple points spread across liver region for robust segmentation"""
        liver_mask = (mask == 1).astype(np.uint8)
        if liver_mask.sum() == 0:
            return None

        y_coords, x_coords = np.where(liver_mask > 0)

        # Get center of mass
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))

        # Get extreme points
        points = [[center_x, center_y]]  # Center

        if num_points >= 3:
            # Add points at quartiles
            points.append([int(np.percentile(x_coords, 25)), int(np.percentile(y_coords, 50))])
            points.append([int(np.percentile(x_coords, 75)), int(np.percentile(y_coords, 50))])

        if num_points >= 5:
            points.append([int(np.percentile(x_coords, 50)), int(np.percentile(y_coords, 25))])
            points.append([int(np.percentile(x_coords, 50)), int(np.percentile(y_coords, 75))])

        return np.array(points[:num_points])

    def segment_slice_with_box(self, ct_slice, box):
        """Segment liver using bounding box prompt"""
        rgb_image = self.prepare_image_for_sam2(ct_slice)
        self.predictor.set_image(rgb_image)

        masks, scores, _ = self.predictor.predict(
            box=box,
            multimask_output=False,
        )
        return masks[0]

    def segment_slice_with_points(self, ct_slice, points):
        """Segment liver using multiple point prompts"""
        rgb_image = self.prepare_image_for_sam2(ct_slice)
        self.predictor.set_image(rgb_image)

        labels = np.ones(len(points), dtype=np.int32)  # All positive points

        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        return masks[scores.argmax()]

    def segment_volume(self, ct_volume, gt_mask=None,
                      start_slice=None, end_slice=None,
                      prompt_type='box'):
        """
        Segment liver from full CT volume using improved prompting

        Args:
            ct_volume: 3D numpy array (H, W, D)
            gt_mask: Ground truth mask for automatic prompts
            start_slice: First slice to process
            end_slice: Last slice to process
            prompt_type: 'box' or 'multipoint'
        """
        h, w, d = ct_volume.shape
        pred_volume = np.zeros((h, w, d), dtype=bool)

        if start_slice is None:
            start_slice = 0
        if end_slice is None:
            end_slice = d

        # Process slices
        for slice_idx in tqdm(range(start_slice, end_slice), desc="Segmenting volume"):
            ct_slice = ct_volume[..., slice_idx]

            # Get prompt from ground truth
            if gt_mask is not None:
                mask_slice = gt_mask[..., slice_idx]

                if prompt_type == 'box':
                    prompt = self.get_liver_bounding_box(mask_slice)
                    if prompt is not None:
                        pred_mask = self.segment_slice_with_box(ct_slice, prompt)
                        pred_volume[..., slice_idx] = pred_mask
                elif prompt_type == 'multipoint':
                    prompt = self.get_liver_multiple_points(mask_slice, num_points=5)
                    if prompt is not None:
                        pred_mask = self.segment_slice_with_points(ct_slice, prompt)
                        pred_volume[..., slice_idx] = pred_mask

        return pred_volume

    @staticmethod
    def compute_dice_score(pred, gt):
        """Compute Dice similarity coefficient"""
        pred = pred.astype(bool)
        gt = (gt > 0).astype(bool)
        intersection = np.logical_and(pred, gt).sum()
        dice = (2.0 * intersection) / (pred.sum() + gt.sum() + 1e-8)
        return dice

    @staticmethod
    def compute_iou(pred, gt):
        """Compute Intersection over Union"""
        pred = pred.astype(bool)
        gt = (gt > 0).astype(bool)
        intersection = np.logical_and(pred, gt).sum()
        union = np.logical_or(pred, gt).sum()
        iou = intersection / (union + 1e-8)
        return iou

    def save_segmentation(self, pred_volume, output_path, affine=None):
        """Save predicted segmentation as NIfTI file"""
        if affine is None:
            affine = np.eye(4)
        pred_nifti = nib.Nifti1Image(pred_volume.astype(np.uint8), affine=affine)
        nib.save(pred_nifti, output_path)
        print(f"Saved segmentation to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Improved Liver Segmentation using SAM2')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input CT volume (.nii or .nii.gz)')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output segmentation (.nii or .nii.gz)')
    parser.add_argument('--gt-mask', type=str, required=True,
                       help='Path to ground truth mask (REQUIRED for automatic prompts)')
    parser.add_argument('--model-cfg', type=str, default='configs/sam2.1/sam2.1_hiera_l.yaml',
                       help='SAM2 model configuration')
    parser.add_argument('--checkpoint', type=str,
                       default='./sam2/checkpoints/sam2.1_hiera_large.pt',
                       help='Path to SAM2 checkpoint')
    parser.add_argument('--start-slice', type=int, default=None,
                       help='First slice to process')
    parser.add_argument('--end-slice', type=int, default=None,
                       help='Last slice to process')
    parser.add_argument('--prompt-type', type=str, default='box',
                       choices=['box', 'multipoint'],
                       help='Prompting strategy: box or multipoint')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cuda', 'mps', 'cpu'],
                       help='Device to use (default: auto-detect)')
    parser.add_argument('--eval', action='store_true',
                       help='Evaluate against ground truth mask')

    args = parser.parse_args()

    # Initialize segmentor
    segmentor = ImprovedLiverSegmentorSAM2(
        model_cfg=args.model_cfg,
        checkpoint=args.checkpoint,
        device=args.device
    )

    # Load input volume
    print(f"\nLoading input volume: {args.input}")
    ct_volume = segmentor.read_nii(args.input)
    print(f"Volume shape: {ct_volume.shape}")

    # Load ground truth mask
    print(f"Loading ground truth mask: {args.gt_mask}")
    gt_mask = segmentor.read_nii(args.gt_mask)
    print(f"GT mask shape: {gt_mask.shape}")

    # Segment volume
    print(f"\nStarting segmentation with '{args.prompt_type}' prompts...")
    pred_volume = segmentor.segment_volume(
        ct_volume,
        gt_mask=gt_mask,
        start_slice=args.start_slice,
        end_slice=args.end_slice,
        prompt_type=args.prompt_type
    )

    print(f"\nSegmentation complete!")
    print(f"Segmented volume shape: {pred_volume.shape}")
    print(f"Number of liver voxels: {pred_volume.sum()}")

    # Save results
    segmentor.save_segmentation(pred_volume, args.output)

    # Evaluate
    if args.eval:
        print("\n" + "="*60)
        print("Evaluation Metrics")
        print("="*60)

        # Overall metrics
        dice = segmentor.compute_dice_score(pred_volume, gt_mask == 1)
        iou = segmentor.compute_iou(pred_volume, gt_mask == 1)

        print(f"Prompt Type: {args.prompt_type.upper()}")
        print(f"Overall Dice Score: {dice:.4f}")
        print(f"Overall IoU: {iou:.4f}")

        # Per-slice metrics
        slice_dice_scores = []
        for i in range(ct_volume.shape[2]):
            if (gt_mask[..., i] == 1).sum() > 0:
                slice_dice = segmentor.compute_dice_score(
                    pred_volume[..., i],
                    gt_mask[..., i] == 1
                )
                slice_dice_scores.append(slice_dice)

        if slice_dice_scores:
            print(f"Mean Slice Dice: {np.mean(slice_dice_scores):.4f} ± {np.std(slice_dice_scores):.4f}")
            print(f"Min Slice Dice: {np.min(slice_dice_scores):.4f}")
            print(f"Max Slice Dice: {np.max(slice_dice_scores):.4f}")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
