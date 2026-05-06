"""
Minimal Bayesian road-vs-background classifier for KITTI grayscale frames.

Assumptions
-----------
- Images are KITTI odometry grayscale frames (image_0) sized 1241x376.
- Road pixels appear mostly in a bottom trapezoid; background near top band.
- Uses 1D intensity histograms with Laplace smoothing to estimate likelihoods.

Usage
-----
    python bayes_road.py \
    --data-root data_odometry_gray/dataset/sequences/09/image_0 \
    --save-frames -1 \
    --out-dir outputs/bayes_road \
    --alpha 0.85

Outputs
-------
- /mask_XXXXXX.png     : binary MAP mask (road=255, bg=0)
- /overlay_XXXXXX.png  : red overlay of road mask on original frame
- /prob_XXXXXX.png     : grayscale probability map (0-255)
"""
#!/usr/bin/env python3
import argparse
import math
from pathlib import Path
import numpy as np
from PIL import Image
import cv2

def list_frames(data_root: Path) -> list[Path]:
    return sorted(data_root.glob("*.png"))

def get_trapezoid_mask(width: int, height: int, vp_y_rate=0.5, bottom_width_rate=0.9, top_width_rate=0.15):
    """
    소실점을 기준으로 도로 가능성이 높은 사다리꼴 마스크를 생성합니다.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    vp_y = int(height * vp_y_rate)
    p1 = [int(width * (0.5 - top_width_rate)), vp_y]
    p2 = [int(width * (0.5 + top_width_rate)), vp_y]
    p3 = [int(width * (0.5 + bottom_width_rate/2)), height]
    p4 = [int(width * (0.5 - bottom_width_rate/2)), height]
    pts = np.array([p1, p2, p3, p4], np.int32)
    cv2.fillPoly(mask, [pts], 1)
    return mask

def accumulate_weighted_histogram(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """마스크 영역 내의 픽셀만 사용하여 히스토그램 생성"""
    pixels = img[mask > 0]
    counts = np.bincount(pixels.flatten(), minlength=256)
    return counts.astype(np.float64)

def classify_frame(img: np.ndarray, road_p: np.ndarray, bg_p: np.ndarray, prior_road: float) -> tuple[np.ndarray, np.ndarray]:
    """Return MAP mask and probability map for road class."""
    eps = 1e-10
    log_p_road = np.log(road_p[img] + eps) + math.log(prior_road)
    log_p_bg   = np.log(bg_p[img]   + eps) + math.log(1 - prior_road)
    logit = log_p_road - log_p_bg
    prob  = 1.0 / (1.0 + np.exp(-np.clip(logit, -15, 15)))
    mask  = logit > 0
    return mask, prob

def run(
    data_root: Path,
    train_frames: int,
    save_frames: int,
    out_dir: Path,
    prior_road: float,
    alpha: float = 0.8,
    video_path: Path | None = None,
    video_fps: float = 10.0,
    vp_y_rate: float = 0.5,
    bottom_width_rate: float = 0.9,
    top_width_rate: float = 0.15,
    max_frames: int | None = None,
    gif_path: Path | None = None,
    gif_fps: float = 8.0,
):
    frames = list_frames(data_root)
    if not frames:
        raise SystemExit(f"No PNG frames found in {data_root}")

    sample_img = np.array(Image.open(frames[0]), dtype=np.uint8)
    h, w = sample_img.shape

    road_mask = get_trapezoid_mask(w, h, vp_y_rate, bottom_width_rate, top_width_rate)
    bg_mask   = 1 - road_mask

    road_p_total = np.ones(256)
    bg_p_total   = np.ones(256)

    out_dir.mkdir(parents=True, exist_ok=True)

    limit          = len(frames) if max_frames is None else min(max_frames, len(frames))
    max_frames_eval = max(train_frames, limit if save_frames >= 0 else len(frames))
    frames_iter    = frames[:max_frames_eval]

    writer = None
    if video_path is not None:
        suffix = video_path.suffix.lower()
        fourcc = cv2.VideoWriter_fourcc(*("mp4v" if suffix == ".mp4" else "MJPG"))
        writer = cv2.VideoWriter(str(video_path), fourcc, video_fps, (w, h))

    gif_frames: list[Image.Image] = []

    for idx, path in enumerate(frames_iter):
        img = np.array(Image.open(path), dtype=np.uint8)

        curr_road_counts = accumulate_weighted_histogram(img, road_mask)
        curr_bg_counts   = accumulate_weighted_histogram(img, bg_mask)

        if idx == 0:
            road_p_total = curr_road_counts + 1
            bg_p_total   = curr_bg_counts   + 1
        else:
            road_p_total = alpha * road_p_total + (1 - alpha) * (curr_road_counts + 1)
            bg_p_total   = alpha * bg_p_total   + (1 - alpha) * (curr_bg_counts   + 1)

        road_p = road_p_total / road_p_total.sum()
        bg_p   = bg_p_total   / bg_p_total.sum()

        should_save = (save_frames < 0 or idx < save_frames or writer is not None) and \
                      (max_frames is None or idx < max_frames)
        if should_save:
            mask, prob = classify_frame(img, road_p, bg_p, prior_road)
            stem = path.stem

            Image.fromarray((mask.astype(np.uint8) * 255)).save(out_dir / f"mask_{stem}.png")
            prob_img = np.clip(prob * 255, 0, 255).astype(np.uint8)
            Image.fromarray(prob_img).save(out_dir / f"prob_{stem}.png")

            overlay = np.stack([img, img, img], axis=-1)
            overlay[mask, 0] = 255
            overlay[mask, 1:] = (overlay[mask, 1:] * 0.3).astype(np.uint8)
            Image.fromarray(overlay).save(out_dir / f"overlay_{stem}.png")

            if writer is not None:
                writer.write(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if gif_path is not None:
                gif_frames.append(Image.fromarray(overlay))

            print(f"Processed {stem} with temporal consistency")

    if writer is not None:
        writer.release()

    if gif_path is not None and gif_frames:
        duration_ms = int(1000 / gif_fps)
        gif_frames[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration_ms,
            loop=0,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",         type=Path,  default=Path("data_odometry_gray/dataset/sequences/09/image_0"))
    parser.add_argument("--train-frames",       type=int,   default=80)
    parser.add_argument("--save-frames",        type=int,   default=-1,   help="frames to save; -1 for all")
    parser.add_argument("--out-dir",            type=Path,  default=Path("outputs/bayes_road"))
    parser.add_argument("--prior-road",         type=float, default=0.5)
    parser.add_argument("--alpha",              type=float, default=0.85,  help="Temporal consistency weight")
    parser.add_argument("--video-path",         type=Path,  default=None,  help="if set, write overlay video here")
    parser.add_argument("--video-fps",          type=float, default=10.0)
    parser.add_argument("--vp-y-rate",          type=float, default=0.5,   help="vanishing point height ratio")
    parser.add_argument("--bottom-width-rate",  type=float, default=0.9)
    parser.add_argument("--top-width-rate",     type=float, default=0.15)
    parser.add_argument("--max-frames",         type=int,   default=None,  help="process at most this many frames")
    parser.add_argument("--gif-path",           type=Path,  default=None,  help="optional GIF output of overlays")
    parser.add_argument("--gif-fps",            type=float, default=8.0)
    args = parser.parse_args()

    run(
        args.data_root,
        args.train_frames,
        args.save_frames,
        args.out_dir,
        args.prior_road,
        args.alpha,
        args.video_path,
        args.video_fps,
        args.vp_y_rate,
        args.bottom_width_rate,
        args.top_width_rate,
        args.max_frames,
        args.gif_path,
        args.gif_fps,
    )