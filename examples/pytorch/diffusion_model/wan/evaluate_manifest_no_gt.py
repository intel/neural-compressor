import argparse
import glob
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.io import wavfile


def _resolve_manifest_path(path_value: str, root: Path) -> Path:
	"""Use absolute paths from manifest directly; fallback to root for relative paths."""
	p = Path(path_value)
	if p.is_absolute():
		return p
	return root / p


def _build_sample_dict(manifest):
	if isinstance(manifest, dict):
		return {str(k): v for k, v in manifest.items() if isinstance(v, dict)}
	if isinstance(manifest, list):
		result = {}
		for idx, item in enumerate(manifest):
			if not isinstance(item, dict):
				continue
			sample_id = str(item.get("id", idx))
			result[sample_id] = item
		return result
	raise ValueError("Manifest must be a JSON object or list")


def build_matched_manifest(source_manifest_path: Path, generated_video_dir: Path):
	with source_manifest_path.open("r", encoding="utf-8") as f:
		source_manifest = json.load(f)

	source_samples = _build_sample_dict(source_manifest)
	manifest_root = source_manifest_path.parent
	video_files = sorted(glob.glob(str(generated_video_dir / "*.mp4")))

	matched = {}
	for sample_id, sample in source_samples.items():
		prompt = sample.get("prompt")
		image = sample.get("image")
		audio = sample.get("audio")
		if not prompt or not image or not audio:
			continue

		if not Path(image).is_absolute():
			image = str((manifest_root / image).resolve())
		if not Path(audio).is_absolute():
			audio = str((manifest_root / audio).resolve())

		prefix = f"{sample_id}_"
		candidates = [vp for vp in video_files if Path(vp).name.startswith(prefix)]
		if not candidates:
			continue

		matched[sample_id] = {
			"prompt": prompt,
			"image": image,
			"audio": audio,
			"generate_video": str(Path(candidates[-1]).resolve()),
		}

	return matched


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
	mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
	if mse <= 1e-12:
		return 100.0
	return 20.0 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
	# Standard single-image SSIM implementation on grayscale images.
	if img1.ndim == 3:
		img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	if img2.ndim == 3:
		img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

	img1 = img1.astype(np.float64)
	img2 = img2.astype(np.float64)

	c1 = (0.01 * 255) ** 2
	c2 = (0.03 * 255) ** 2

	kernel = cv2.getGaussianKernel(11, 1.5)
	window = kernel @ kernel.T

	mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
	mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
	mu1_sq = mu1 * mu1
	mu2_sq = mu2 * mu2
	mu1_mu2 = mu1 * mu2

	sigma1_sq = cv2.filter2D(img1 * img1, -1, window)[5:-5, 5:-5] - mu1_sq
	sigma2_sq = cv2.filter2D(img2 * img2, -1, window)[5:-5, 5:-5] - mu2_sq
	sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

	ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
		(mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
	)
	return float(ssim_map.mean())


def read_video_frames(video_path: Path, max_frames: int = 120):
	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise RuntimeError(f"Failed to open video: {video_path}")

	frames = []
	while cap.isOpened() and len(frames) < max_frames:
		ok, frame = cap.read()
		if not ok:
			break
		frames.append(frame)

	cap.release()
	if not frames:
		raise RuntimeError(f"No frame read from video: {video_path}")
	return frames


def _read_audio_mono(audio_path: Path):
	sr, wav = wavfile.read(str(audio_path))
	if wav.ndim == 2:
		wav = wav.mean(axis=1)
	wav = wav.astype(np.float32)
	if wav.dtype != np.float32:
		max_abs = np.max(np.abs(wav)) if np.max(np.abs(wav)) > 0 else 1.0
		wav = wav / max_abs
	return wav, int(sr)


def _resample_1d(x: np.ndarray, target_len: int):
	if len(x) == target_len:
		return x
	if len(x) == 0:
		return np.zeros(target_len, dtype=np.float32)
	xp = np.linspace(0.0, 1.0, num=len(x), endpoint=True)
	xnew = np.linspace(0.0, 1.0, num=target_len, endpoint=True)
	return np.interp(xnew, xp, x).astype(np.float32)


def _face_mouth_roi(gray: np.ndarray, face_cascade):
	faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
	if len(faces) > 0:
		x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
		x1 = int(x + 0.2 * w)
		x2 = int(x + 0.8 * w)
		y1 = int(y + 0.55 * h)
		y2 = int(y + 0.95 * h)
	else:
		h, w = gray.shape[:2]
		x1, x2 = int(0.3 * w), int(0.7 * w)
		y1, y2 = int(0.58 * h), int(0.9 * h)
	x1 = max(0, min(x1, gray.shape[1] - 1))
	x2 = max(x1 + 1, min(x2, gray.shape[1]))
	y1 = max(0, min(y1, gray.shape[0] - 1))
	y2 = max(y1 + 1, min(y2, gray.shape[0]))
	return x1, y1, x2, y2


def compute_sync_c(video_frames, audio_path: Path, fps: float = 25.0):
	# Proxy Sync-C: correlation between mouth-region motion and audio energy envelope.
	try:
		audio, sr = _read_audio_mono(audio_path)
		if len(video_frames) < 2 or len(audio) < 2:
			return 0.0

		samples_per_frame = max(1, int(sr / fps))
		# Use per-frame audio energy then align with frame-difference count.
		n_audio_frames = max(1, len(audio) // samples_per_frame)
		audio = audio[: n_audio_frames * samples_per_frame]
		audio_frame = audio.reshape(n_audio_frames, samples_per_frame)
		audio_energy = np.mean(np.abs(audio_frame), axis=1)

		face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
		motions = []
		prev = cv2.cvtColor(video_frames[0], cv2.COLOR_BGR2GRAY)
		for fr in video_frames[1:]:
			cur = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
			x1, y1, x2, y2 = _face_mouth_roi(cur, face_cascade)
			d = np.mean(np.abs(cur[y1:y2, x1:x2].astype(np.float32) - prev[y1:y2, x1:x2].astype(np.float32)))
			motions.append(d)
			prev = cur

		motions = np.asarray(motions, dtype=np.float32)
		if len(motions) < 2:
			return 0.0
		audio_aligned = _resample_1d(audio_energy.astype(np.float32), len(motions))
		if np.std(audio_aligned) < 1e-8 or np.std(motions) < 1e-8:
			return 0.0
		corr = np.corrcoef(audio_aligned, motions)[0, 1]
		if np.isnan(corr):
			return 0.0
		return float(np.clip(corr, -1.0, 1.0))
	except Exception:
		return 0.0


def compute_hkc_hkv(video_frames):
	# Proxy HKC/HKV using side-region hand-motion energy statistics.
	if len(video_frames) < 2:
		return 0.0, 0.0

	prev = cv2.cvtColor(video_frames[0], cv2.COLOR_BGR2GRAY)
	hand_motion = []
	for fr in video_frames[1:]:
		cur = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
		h, w = cur.shape
		# two side regions where hands often appear in portrait talking videos
		left = (slice(int(0.35 * h), int(0.85 * h)), slice(0, int(0.32 * w)))
		right = (slice(int(0.35 * h), int(0.85 * h)), slice(int(0.68 * w), w))
		dleft = np.mean(np.abs(cur[left].astype(np.float32) - prev[left].astype(np.float32)))
		dright = np.mean(np.abs(cur[right].astype(np.float32) - prev[right].astype(np.float32)))
		hand_motion.append((dleft + dright) * 0.5)
		prev = cur

	hm = np.asarray(hand_motion, dtype=np.float32)
	if hm.size == 0:
		return 0.0, 0.0
	# HKC proxy in [0,1]: normalized average hand activity.
	hkc = float(np.clip(hm.mean() / 25.0, 0.0, 1.0))
	hkv = float(np.var(hm))
	return hkc, hkv


def compute_csim(reference_bgr: np.ndarray, target_bgr: np.ndarray):
	# Proxy CSIM: cosine similarity of color+texture descriptor.
	def feat(img):
		img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		hist = cv2.calcHist([hsv], [0, 1], None, [24, 24], [0, 180, 0, 256]).flatten().astype(np.float32)
		hist = hist / (np.linalg.norm(hist) + 1e-8)
		g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		edges = cv2.Canny(g, 60, 160).astype(np.float32)
		edges = cv2.resize(edges, (56, 56), interpolation=cv2.INTER_AREA).flatten()
		edges = edges / (np.linalg.norm(edges) + 1e-8)
		return np.concatenate([hist, edges], axis=0)

	f1 = feat(reference_bgr)
	f2 = feat(target_bgr)
	sim = float(np.dot(f1, f2) / ((np.linalg.norm(f1) * np.linalg.norm(f2)) + 1e-8))
	return float(np.clip(sim, -1.0, 1.0))


def _feature_for_efid(img_bgr: np.ndarray):
	img = cv2.resize(img_bgr, (224, 224), interpolation=cv2.INTER_AREA)
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	hist_hs = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256]).flatten().astype(np.float64)
	hist_hs = hist_hs / (np.sum(hist_hs) + 1e-12)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0
	stat = np.array([gray.mean(), gray.std()], dtype=np.float64)
	return np.concatenate([hist_hs, stat], axis=0)


def _sqrtm_psd(mat: np.ndarray):
	# Symmetric PSD matrix square root via eigen decomposition.
	vals, vecs = np.linalg.eigh(mat)
	vals = np.clip(vals, 0.0, None)
	return (vecs * np.sqrt(vals)) @ vecs.T


def frechet_distance(feats1: np.ndarray, feats2: np.ndarray):
	mu1 = np.mean(feats1, axis=0)
	mu2 = np.mean(feats2, axis=0)
	s1 = np.cov(feats1, rowvar=False)
	s2 = np.cov(feats2, rowvar=False)
	if s1.ndim == 0:
		s1 = np.array([[float(s1)]])
	if s2.ndim == 0:
		s2 = np.array([[float(s2)]])

	covmean = _sqrtm_psd(s1 @ s2)
	diff = mu1 - mu2
	fid = diff @ diff + np.trace(s1 + s2 - 2.0 * covmean)
	return float(max(fid, 0.0))


def evaluate_manifest(
	manifest_path: Path,
	output_path: Path,
	max_frames: int,
	metric_size: int,
):
	with manifest_path.open("r", encoding="utf-8") as f:
		manifest = json.load(f)
	manifest_dir = manifest_path.parent

	if not isinstance(manifest, dict) or not manifest:
		raise ValueError("Manifest must be a non-empty object.")

	per_sample = {}
	ssim_first_list = []
	psnr_first_list = []
	ssim_avg_frames_list = []
	psnr_avg_frames_list = []
	sync_c_list = []
	hkc_list = []
	hkv_list = []
	csim_list = []
	efid_ref_feats = []
	efid_gen_feats = []
	failed = []

	for sample_id, sample in manifest.items():
		try:
			image_rel = sample.get("image")
			video_rel = sample.get("generate_video")
			audio_rel = sample.get("audio")
			if not image_rel or not video_rel:
				raise ValueError("Missing image or generate_video field")

			image_path = _resolve_manifest_path(image_rel, manifest_dir)
			video_path = _resolve_manifest_path(video_rel, manifest_dir)
			audio_path = _resolve_manifest_path(audio_rel, manifest_dir) if audio_rel else None

			if not image_path.exists():
				raise FileNotFoundError(f"Image not found: {image_path}")
			if not video_path.exists():
				raise FileNotFoundError(f"Video not found: {video_path}")

			ref = cv2.imread(str(image_path))
			if ref is None:
				raise RuntimeError(f"Cannot read image: {image_path}")

			frames = read_video_frames(video_path, max_frames=max_frames)

			# Resize for faster and more stable metric computation.
			ref_m = cv2.resize(ref, (metric_size, metric_size), interpolation=cv2.INTER_AREA)
			frames_m = [cv2.resize(fr, (metric_size, metric_size), interpolation=cv2.INTER_AREA) for fr in frames]

			first = frames_m[0]
			ssim_first = ssim(ref_m, first)
			psnr_first = psnr(ref_m, first)
			sync_c = compute_sync_c(frames_m, audio_path, fps=25.0) if audio_path else 0.0
			hkc, hkv = compute_hkc_hkv(frames_m)
			csim = compute_csim(ref_m, first)

			ssim_frames = []
			psnr_frames = []
			for fr in frames_m:
				ssim_frames.append(ssim(ref_m, fr))
				psnr_frames.append(psnr(ref_m, fr))

			ssim_avg = float(np.mean(ssim_frames))
			psnr_avg = float(np.mean(psnr_frames))

			per_sample[sample_id] = {
				"image": image_rel,
				"generate_video": video_rel,
				"num_frames_used": len(frames),
				"ssim_image_vs_first_frame": ssim_first,
				"psnr_image_vs_first_frame": psnr_first,
				"ssim_image_vs_all_frames_avg": ssim_avg,
				"psnr_image_vs_all_frames_avg": psnr_avg,
				"Sync-C": sync_c,
				"HKC": hkc,
				"HKV": hkv,
				"CSIM": csim,
			}

			ssim_first_list.append(ssim_first)
			psnr_first_list.append(psnr_first)
			ssim_avg_frames_list.append(ssim_avg)
			psnr_avg_frames_list.append(psnr_avg)
			sync_c_list.append(sync_c)
			hkc_list.append(hkc)
			hkv_list.append(hkv)
			csim_list.append(csim)

			efid_ref_feats.append(_feature_for_efid(ref_m))
			efid_gen_feats.append(_feature_for_efid(first))
		except Exception as e:
			failed.append({"sample_id": sample_id, "error": str(e)})

	efid = None
	if len(efid_ref_feats) >= 2 and len(efid_gen_feats) >= 2:
		efid = frechet_distance(np.stack(efid_ref_feats, axis=0), np.stack(efid_gen_feats, axis=0))

	summary = {
		"num_samples_total": len(manifest),
		"num_samples_success": len(per_sample),
		"num_samples_failed": len(failed),
		"metrics": {
			"ssim_image_vs_first_frame_mean": float(np.mean(ssim_first_list)) if ssim_first_list else None,
			"psnr_image_vs_first_frame_mean": float(np.mean(psnr_first_list)) if psnr_first_list else None,
			"ssim_image_vs_all_frames_avg_mean": float(np.mean(ssim_avg_frames_list)) if ssim_avg_frames_list else None,
			"psnr_image_vs_all_frames_avg_mean": float(np.mean(psnr_avg_frames_list)) if psnr_avg_frames_list else None,
			"Sync-C_mean": float(np.mean(sync_c_list)) if sync_c_list else None,
			"HKC_mean": float(np.mean(hkc_list)) if hkc_list else None,
			"HKV_mean": float(np.mean(hkv_list)) if hkv_list else None,
			"CSIM_mean": float(np.mean(csim_list)) if csim_list else None,
			"EFID_reference_vs_firstframe": efid,
		},
		"unavailable_metrics": {
			"FID": "Unavailable without real image/video distribution (ground-truth set).",
			"FVD": "Unavailable without real video set and feature extractor pipeline for real vs generated distributions.",
			"reason": "Current manifest contains prompt/image/audio/generate_video but no real video references.",
		},
		"metric_notes": {
			"Sync-C": "No-GT proxy via audio-energy and mouth-motion correlation.",
			"HKC": "No-GT proxy via side-region hand-motion confidence.",
			"HKV": "No-GT proxy via variance of hand-motion energy.",
			"CSIM": "No-GT proxy identity similarity using color+texture cosine similarity.",
			"EFID": "No-GT proxy Fréchet distance between reference-image and generated-first-frame handcrafted features.",
		},
	}

	output = {
		"config": {
			"manifest_path": str(manifest_path),
				"path_resolution": "Use absolute paths in manifest directly; relative paths are resolved by manifest dir.",
			"max_frames": max_frames,
			"metric_size": metric_size,
			"note": "This is no-ground-truth evaluation. SSIM/PSNR are computed as image-to-video fidelity proxies.",
		},
		"summary": summary,
		"per_sample": per_sample,
		"failed_samples": failed,
	}

	with output_path.open("w", encoding="utf-8") as f:
		json.dump(output, f, ensure_ascii=False, indent=2)

	return output


def main():
	parser = argparse.ArgumentParser(description="No-GT evaluation for S2V manifest")
	parser.add_argument(
		"--manifest",
		type=str,
		default="./s2v_manifest_with_generate_video.json",
		help="Input manifest JSON path. Default: ./s2v_manifest_with_generate_video.json",
	)
	parser.add_argument(
		"--output",
		type=str,
		default="./evaluation_no_gt_metrics_s2v.json",
		help="Output metrics JSON path. Default: ./evaluation_no_gt_metrics_s2v.json",
	)
	parser.add_argument(
		"--generated_video_dir",
		type=str,
		default=None,
		help="Directory containing generated mp4 files. If set, evaluator will build matched manifest from --manifest.",
	)
	parser.add_argument(
		"--matched_manifest_output",
		type=str,
		default=None,
		help="Output path for matched manifest with generate_video field.",
	)
	parser.add_argument("--max_frames", type=int, default=120)
	parser.add_argument("--metric_size", type=int, default=256)
	args = parser.parse_args()

	manifest_path = Path(args.manifest)
	if args.generated_video_dir:
		matched = build_matched_manifest(
			source_manifest_path=manifest_path,
			generated_video_dir=Path(args.generated_video_dir),
		)
		matched_manifest_output = Path(args.matched_manifest_output) if args.matched_manifest_output else Path(
			"./s2v_manifest_with_generate_video.json"
		)
		matched_manifest_output.parent.mkdir(parents=True, exist_ok=True)
		matched_manifest_output.write_text(json.dumps(matched, ensure_ascii=False, indent=2), encoding="utf-8")
		manifest_path = matched_manifest_output

	out = evaluate_manifest(
		manifest_path=manifest_path,
		output_path=Path(args.output),
		max_frames=args.max_frames,
		metric_size=args.metric_size,
	)

	print("Evaluation done.")
	print(json.dumps(out["summary"], ensure_ascii=False, indent=2))
	print(f"Saved: {args.output}")


if __name__ == "__main__":
	main()
