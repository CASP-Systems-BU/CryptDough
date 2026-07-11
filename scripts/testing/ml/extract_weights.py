#!/usr/bin/env python3
"""
extract_weights.py - Extract pretrained model weights and dataset instances
into fixed-point binary files for cdough.

Models:
  - AlexNet (ImageNet)
  - VGG16-CIFAR10 (CIFAR-10)
  - VGG16-ImageNet (ImageNet)

Instances:
  - ImageNet-like images (for VGG16-ImageNet inference)

Usage:
    pip install torch torchvision numpy pillow
    python extract_weights.py                    # weights only
    python extract_weights.py --instances        # weights + instances
    python extract_weights.py --instances-only   # instances only
"""

import os
import sys
import json
import struct
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

PRECISION = 12
INT_TYPE = "int32"
OUTPUT_DIR = "weights"
INSTANCES_DIR = "instances"

DTYPE_MAP = {
    "int32": (np.int32, "<i"),
    "int64": (np.int64, "<q"),
}


# conversion helpers
def float_to_fixed_point(tensor, precision, dtype):
    np_dtype, _ = DTYPE_MAP[dtype]
    arr = tensor.detach().cpu().numpy().astype(np.float64)
    scale = float(1 << precision)
    scaled = np.round(arr * scale)

    if np_dtype == np.int32:
        lo, hi = -(2**31), 2**31 - 1
    else:
        lo, hi = -(2**63), 2**63 - 1

    overflow = np.sum((scaled > hi) | (scaled < lo))
    if overflow > 0:
        print(f"  WARNING: {overflow}/{scaled.size} values overflow {dtype}!")
        scaled = np.clip(scaled, lo, hi)

    return scaled.flatten().astype(np_dtype)


def save_binary(data, filepath, dtype):
    _, fmt = DTYPE_MAP[dtype]
    with open(filepath, "wb") as f:
        for val in data:
            f.write(struct.pack(fmt, int(val)))
    expected = len(data) * (4 if dtype == "int32" else 8)
    assert os.path.getsize(filepath) == expected


def generate_verification(original_tensor, fixed_array, precision, num_samples=10):
    flat = original_tensor.detach().cpu().numpy().flatten()
    scale = float(1 << precision)
    indices = np.linspace(0, len(flat) - 1, num_samples, dtype=int)

    lines = [f"{'Index':>8} | {'Original':>12} | {'Fixed-Point':>12} | {'Reconstructed':>14} | {'Error':>12}"]
    lines.append("-" * 75)
    for idx in indices:
        orig, fixed = flat[idx], fixed_array[idx]
        recon = fixed / scale
        lines.append(f"{idx:>8d} | {orig:>12.6f} | {fixed:>12d} | {recon:>14.6f} | {abs(orig - recon):>12.8f}")

    recon_all = fixed_array.astype(np.float64) / scale
    err = np.abs(flat - recon_all)
    lines += ["", f"Max absolute error:  {err.max():.8f}", f"Mean absolute error: {err.mean():.8f}"]
    return "\n".join(lines)


# extract weights 
def extract_model(model, model_name, precision, dtype, output_dir):
    model.eval()
    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    manifest = {
        "model_name": model_name, "precision": precision,
        "int_type": dtype, "scale_factor": 1 << precision, "layers": []
    }
    verification_lines = [
        f"Model: {model_name}",
        f"Precision: {precision} bits (scale factor: {1 << precision})",
        f"Integer type: {dtype}", "=" * 75, ""
    ]

    print(f"\n{'=' * 60}")
    print(f"Extracting: {model_name}")
    print(f"Output: {model_dir}/")
    print(f"{'=' * 60}")

    for name, module in model.named_modules():
        if len(list(module.children())) > 0:
            continue

        layer_info = {"name": name, "type": type(module).__name__, "params": []}
        has_params = False

        if hasattr(module, 'weight') and module.weight is not None:
            has_params = True
            weight = module.weight.data
            shape = list(weight.shape)

            if isinstance(module, torch.nn.Conv2d):
                weight_for_export = weight.permute(0, 2, 3, 1).contiguous()
            else:
                weight_for_export = weight

            fixed = float_to_fixed_point(weight_for_export, precision, dtype)
            filename = f"{name.replace('.', '_')}_weight.bin"
            save_binary(fixed, os.path.join(model_dir, filename), dtype)

            layer_info["params"].append({
                "param": "weight", "shape": shape,
                "num_elements": int(np.prod(shape)), "file": filename,
                "original_range": [float(weight.min()), float(weight.max())]
            })
            print(f"  {name:40s} weight  {str(shape):20s} -> {filename}")

            verification_lines.append(f"Layer: {name} -- weight {shape}")
            verification_lines.append(generate_verification(weight, fixed, precision))
            verification_lines.append("")

        if hasattr(module, 'bias') and module.bias is not None:
            has_params = True
            bias = module.bias.data
            shape = list(bias.shape)
            fixed = float_to_fixed_point(bias, precision, dtype)
            filename = f"{name.replace('.', '_')}_bias.bin"
            save_binary(fixed, os.path.join(model_dir, filename), dtype)
            layer_info["params"].append({
                "param": "bias", "shape": shape,
                "num_elements": int(np.prod(shape)), "file": filename,
                "original_range": [float(bias.min()), float(bias.max())]
            })
            print(f"  {name:40s} bias    {str(shape):20s} -> {filename}")

        if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            for attr_name in ['running_mean', 'running_var']:
                attr = getattr(module, attr_name)
                if attr is not None:
                    has_params = True
                    shape = list(attr.shape)
                    fixed = float_to_fixed_point(attr, precision, dtype)
                    filename = f"{name.replace('.', '_')}_{attr_name}.bin"
                    save_binary(fixed, os.path.join(model_dir, filename), dtype)
                    layer_info["params"].append({
                        "param": attr_name, "shape": shape,
                        "num_elements": int(np.prod(shape)), "file": filename,
                        "original_range": [float(attr.min()), float(attr.max())]
                    })
                    print(f"  {name:40s} {attr_name:8s}{str(shape):20s} -> {filename}")
            layer_info["epsilon"] = module.eps
            layer_info["momentum"] = module.momentum

        if has_params:
            manifest["layers"].append(layer_info)

    with open(os.path.join(model_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    with open(os.path.join(model_dir, "verification.txt"), "w") as f:
        f.write("\n".join(verification_lines))

    total = sum(p["num_elements"] for l in manifest["layers"] for p in l["params"])
    bpe = 4 if dtype == "int32" else 8
    print(f"  Total: {total:,} params, {total * bpe / 1024 / 1024:.1f} MB")


# extract instances

def download_image(url, filepath):
    """Download an image with a User-Agent header (Wikimedia blocks bare requests)."""
    from urllib.request import urlopen, Request
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req) as resp, open(filepath, "wb") as f:
        f.write(resp.read())


def image_to_cdough_bin(tensor_chw, precision, dtype):
    """
    Convert a preprocessed [C, H, W] tensor to cdough's channel-interleaved
    fixed-point format: [H, W*C] where each row has pixels with channels
    interleaved (pixel0_ch0, pixel0_ch1, pixel0_ch2, pixel1_ch0, ...).
    """
    C, H, W = tensor_chw.shape
    np_dtype, _ = DTYPE_MAP[dtype]
    arr = tensor_chw.numpy().astype(np.float64)
    scale = float(1 << precision)

    # build [H, W*C] interleaved layout
    cdough = np.zeros((H, W * C), dtype=np.float64)
    for h in range(H):
        for w in range(W):
            for c in range(C):
                cdough[h, w * C + c] = arr[c, h, w]

    scaled = np.round(cdough * scale).flatten()

    if np_dtype == np.int32:
        lo, hi = -(2**31), 2**31 - 1
    else:
        lo, hi = -(2**63), 2**63 - 1

    return np.clip(scaled, lo, hi).astype(np_dtype)


def extract_imagenet_instances(output_dir, precision, dtype, num_images=5):
    """
    Download sample images, preprocess with VGG16's standard ImageNet
    normalization, convert to cdough format, and save as .bin files.
    Also runs PyTorch VGG16 to get expected top-5 predictions.
    """
    from PIL import Image

    inst_dir = os.path.join(output_dir, "imagenet")
    os.makedirs(inst_dir, exist_ok=True)

    # public domain / CC-licensed test images
    sample_images = [
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/YellowLabradorLooking_new.jpg/1200px-YellowLabradorLooking_new.jpg", "labrador"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg", "tabby_cat"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/1200px-Camponotus_flavomarginatus_ant.jpg", "ant"),
        ("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg", "cat"),
    ]

    # standard VGG16 ImageNet preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    print("\nLoading VGG16 for reference predictions...")
    model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
    model.eval()

    manifest = {"precision": precision, "format": "[H, W*C] channel-interleaved", "images": []}

    print(f"\n{'=' * 60}")
    print(f"Extracting ImageNet instances")
    print(f"Output: {inst_dir}/")
    print(f"{'=' * 60}")

    for i, (url, name) in enumerate(sample_images[:num_images]):
        print(f"\n  [{i}] {name}")
        img_path = os.path.join(inst_dir, f"{name}.jpg")

        try:
            download_image(url, img_path)
        except Exception as e:
            print(f"    Failed to download: {e}")
            continue

        img = Image.open(img_path).convert("RGB")
        tensor = preprocess(img)  # [3, 224, 224]

        # PyTorch reference inference
        with torch.no_grad():
            logits = model(tensor.unsqueeze(0))  # [1, 1000]
            probs = torch.softmax(logits, dim=1)
            top5 = torch.topk(probs, 5)

        top5_cls = top5.indices[0].tolist()
        top5_prob = top5.values[0].tolist()

        # get class labels from torchvision
        categories = torchvision.models.VGG16_Weights.DEFAULT.meta["categories"]
        print(f"    PyTorch top-5:")
        for rank, (cls, prob) in enumerate(zip(top5_cls, top5_prob)):
            print(f"      {rank+1}. {categories[cls]} (class {cls}): {prob:.4f}")

        # save as cdough .bin
        cdough_data = image_to_cdough_bin(tensor, precision, dtype)
        bin_path = os.path.join(inst_dir, f"{name}.bin")
        save_binary(cdough_data, bin_path, dtype)
        print(f"    -> {bin_path} ({len(cdough_data)} values, {224}x{224*3} = {224*224*3})")

        manifest["images"].append({
            "name": name,
            "file": f"{name}.bin",
            "rows": 224,
            "cols": 224 * 3,
            "top5_classes": top5_cls,
            "top5_probs": [round(p, 6) for p in top5_prob],
            "top5_labels": [categories[c] for c in top5_cls],
        })

    with open(os.path.join(inst_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  {len(manifest['images'])} images saved to {inst_dir}/")

def main():
    do_weights = "--instances-only" not in sys.argv
    do_instances = "--instances" in sys.argv or "--instances-only" in sys.argv

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if do_weights:
        print("Loading pretrained models...\n")

        print("Loading AlexNet (ImageNet pretrained)...")
        alexnet = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.DEFAULT)
        extract_model(alexnet, "alexnet_imagenet", PRECISION, INT_TYPE, OUTPUT_DIR)

        print("\nLoading VGG16-BN (CIFAR-10 pretrained)...")
        vgg16_cifar10 = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn",
            pretrained=True, trust_repo=True
        )
        extract_model(vgg16_cifar10, "vgg16_cifar10", PRECISION, INT_TYPE, OUTPUT_DIR)

        print("\nLoading VGG16 (ImageNet pretrained)...")
        vgg16_imagenet = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)
        extract_model(vgg16_imagenet, "vgg16_imagenet", PRECISION, INT_TYPE, OUTPUT_DIR)

        print(f"\n{'=' * 60}")
        print(f"Weights saved to {OUTPUT_DIR}/")
        print(f"{'=' * 60}")

    if do_instances:
        os.makedirs(INSTANCES_DIR, exist_ok=True)
        extract_imagenet_instances(INSTANCES_DIR, PRECISION, INT_TYPE)

    if not do_weights and not do_instances:
        print("Usage:")
        print("  python extract_weights.py                  # weights only")
        print("  python extract_weights.py --instances       # weights + instances")
        print("  python extract_weights.py --instances-only  # instances only")


if __name__ == "__main__":
    main()