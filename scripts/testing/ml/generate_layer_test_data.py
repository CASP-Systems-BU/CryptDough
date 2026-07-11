#!/usr/bin/env python3
"""
generate_layer_test_data.py — Generate ground-truth test data for cdough layer verification

   1:  precision=0, tiny BASELINE
            3x3 in, 2ch->2ch, 2x2 filt, stride=1, pad=0

   1b: precision=0, ISOLATION tests — change ONE variable at a time:
            A: only inCh 2->3
            B: only outCh 2->4
            C: only filter 2x2->3x3
            D: only padding 0->1
            E: only input size 3x3->8x8
            F: everything medium EXCEPT pad=0

   2:  precision=0, medium
            8x8 in, 3ch->4ch, 3x3 filt, stride=1, pad=1

   3-5: increasing precision (fixed-point tests)

Usage:
    python generate_layer_test_data.py
"""

import os
import struct
import numpy as np
import torch
import torch.nn.functional as F

OUTPUT_DIR = "test_data"


def save_bin_int32(data, filepath):
    flat = data.flatten().astype(np.int32)
    with open(filepath, "wb") as f:
        for val in flat:
            f.write(struct.pack("<i", int(val)))
    print(f"    Saved {filepath} ({len(flat)} values)")


def save_bin_int64(data, filepath):
    flat = data.flatten().astype(np.int64)
    with open(filepath, "wb") as f:
        for val in flat:
            f.write(struct.pack("<q", int(val)))
    print(f"    Saved {filepath} ({len(flat)} values)")


def generate_fc_test(test_dir, inputDim, outputDim, precision, label):
    print(f"\n  --- FC: {label} ---")
    print(f"  inputDim={inputDim}, outputDim={outputDim}, precision={precision}")

    os.makedirs(test_dir, exist_ok=True)
    scale = 1 << precision

    np.random.seed(42)
    weight_int = np.random.randint(-3, 4, size=(outputDim, inputDim))
    bias_int = np.random.randint(-5, 6, size=(outputDim,))
    input_int = np.random.randint(-4, 5, size=(1, inputDim))

    weight_pt = torch.tensor(weight_int, dtype=torch.float64)
    bias_pt = torch.tensor(bias_int, dtype=torch.float64)
    input_pt = torch.tensor(input_int, dtype=torch.float64)
    output_pt = F.linear(input_pt, weight_pt, bias_pt)

    if precision == 0:
        cdough_weight = weight_int.flatten().astype(np.int32)
        cdough_bias = bias_int.flatten().astype(np.int32)
        cdough_input = input_int.flatten().astype(np.int32)
        expected = output_pt.detach().numpy().flatten().astype(np.int64)
    else:
        cdough_weight = np.round(weight_int.astype(np.float64) * scale).astype(np.int32)
        cdough_bias = np.round(bias_int.astype(np.float64) * scale).astype(np.int32)
        cdough_input = np.round(input_int.astype(np.float64) * scale).astype(np.int32)
        expected = np.round(output_pt.detach().numpy().flatten() * scale).astype(np.int64)

    save_bin_int32(cdough_weight, os.path.join(test_dir, "fc_weight.bin"))
    save_bin_int32(cdough_bias, os.path.join(test_dir, "fc_bias.bin"))
    save_bin_int32(cdough_input, os.path.join(test_dir, "fc_input.bin"))
    save_bin_int64(expected, os.path.join(test_dir, "fc_expected_output.bin"))

    if inputDim <= 8:
        print(f"  Input:    {input_int.flatten().tolist()}")
        print(f"  Weight:   {weight_int.tolist()}")
        print(f"  Bias:     {bias_int.tolist()}")
        print(f"  PyTorch:  {output_pt.detach().numpy().flatten().tolist()}")
        print(f"  Expected: {expected.tolist()}")

    with open(os.path.join(test_dir, "fc_meta.txt"), "w") as f:
        f.write(f"inputDim={inputDim}\n")
        f.write(f"outputDim={outputDim}\n")
        f.write(f"precision={precision}\n")
        f.write(f"label={label}\n")


def generate_conv2d_test(test_dir, inputH, inputW, inChannels, outChannels,
                          kH, kW, stride, padding, precision, label):
    print(f"\n  --- Conv2D: {label} ---")
    print(f"  input={inputH}x{inputW}, inCh={inChannels}, outCh={outChannels}, "
          f"filter={kH}x{kW}, stride={stride}, pad={padding}, prec={precision}")

    os.makedirs(test_dir, exist_ok=True)
    scale = 1 << precision

    np.random.seed(123)
    weight_int = np.random.randint(-3, 4, size=(outChannels, inChannels, kH, kW))
    input_int = np.random.randint(-4, 5, size=(1, inChannels, inputH, inputW))

    weight_pt = torch.tensor(weight_int, dtype=torch.float64)
    input_pt = torch.tensor(input_int, dtype=torch.float64)
    output_pt = F.conv2d(input_pt, weight_pt, bias=None, stride=stride, padding=padding)
    output_np = output_pt.detach().numpy()

    outH = (inputH + 2 * padding - kH) // stride + 1
    outW = (inputW + 2 * padding - kW) // stride + 1

    print(f"  Output shape: [1, {outChannels}, {outH}, {outW}]")

    # cdough input: [H, W*inCh] interleaved
    cdough_input = np.zeros((inputH, inputW * inChannels), dtype=np.int32)
    for h in range(inputH):
        for w in range(inputW):
            for c in range(inChannels):
                val = float(input_int[0, c, h, w])
                if precision > 0:
                    val = round(val * scale)
                cdough_input[h, w * inChannels + c] = int(val)

    # cdough filter: [outCh*kH, kW*inCh] interleaved
    cdough_filter = np.zeros((outChannels * kH, kW * inChannels), dtype=np.int32)
    for o in range(outChannels):
        for fh in range(kH):
            row_idx = o * kH + fh
            for fw in range(kW):
                for c in range(inChannels):
                    col_idx = fw * inChannels + c
                    val = float(weight_int[o, c, fh, fw])
                    if precision > 0:
                        val = round(val * scale)
                    cdough_filter[row_idx, col_idx] = int(val)

    # cdough output: [outH, outW*outCh] interleaved
    cdough_output = np.zeros((outH, outW * outChannels), dtype=np.int64)
    for h in range(outH):
        for w in range(outW):
            for c in range(outChannels):
                val = float(output_np[0, c, h, w])
                if precision > 0:
                    val = round(val * scale)
                cdough_output[h, w * outChannels + c] = int(val)

    if inputH <= 4 and inputW <= 4:
        print(f"  cdough input ({inputH}x{inputW * inChannels}):")
        for r in range(inputH):
            print(f"    row {r}: {cdough_input[r].tolist()}")
        print(f"  cdough filter ({outChannels * kH}x{kW * inChannels}):")
        for r in range(outChannels * kH):
            print(f"    row {r}: {cdough_filter[r].tolist()}")
        print(f"  cdough expected output ({outH}x{outW * outChannels}):")
        for r in range(outH):
            print(f"    row {r}: {cdough_output[r].tolist()}")

    save_bin_int32(cdough_input, os.path.join(test_dir, "conv_input.bin"))
    save_bin_int32(cdough_filter, os.path.join(test_dir, "conv_filter.bin"))
    save_bin_int64(cdough_output, os.path.join(test_dir, "conv_expected_output.bin"))

    with open(os.path.join(test_dir, "conv_meta.txt"), "w") as f:
        f.write(f"inChannels={inChannels}\n")
        f.write(f"outChannels={outChannels}\n")
        f.write(f"kH={kH}\n")
        f.write(f"kW={kW}\n")
        f.write(f"inputH={inputH}\n")
        f.write(f"inputW={inputW}\n")
        f.write(f"stride={stride}\n")
        f.write(f"padding={padding}\n")
        f.write(f"precision={precision}\n")
        f.write(f"outH={outH}\n")
        f.write(f"outW={outW}\n")
        f.write(f"cdough_input_rows={inputH}\n")
        f.write(f"cdough_input_cols={inputW * inChannels}\n")
        f.write(f"cdough_filter_rows={outChannels * kH}\n")
        f.write(f"cdough_filter_cols={kW * inChannels}\n")
        f.write(f"cdough_output_rows={outH}\n")
        f.write(f"cdough_output_cols={outW * outChannels}\n")
        f.write(f"label={label}\n")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ================================================================
    # PHASE 1: precision=0, tiny BASELINE (known to PASS)
    #   Baseline: 3x3 in, 2 inCh, 2 outCh, 2x2 filt, stride=1, pad=0
    # ================================================================
    print("=" * 60)
    print("PHASE 1: precision=0, tiny BASELINE (known to PASS)")
    print("  3x3 in, 2ch->2ch, 2x2 filt, stride=1, pad=0")
    print("=" * 60)

    generate_fc_test(
        os.path.join(OUTPUT_DIR, "fc_p0_tiny"),
        inputDim=4, outputDim=3, precision=0,
        label="FC p0 4x3")

    generate_conv2d_test(
        os.path.join(OUTPUT_DIR, "conv_p0_tiny"),
        inputH=3, inputW=3, inChannels=2, outChannels=2,
        kH=2, kW=2, stride=1, padding=0, precision=0,
        label="BASELINE: 3x3 2ch->2ch 2x2filt pad=0")


    # change ONE variable from baseline
    # Change one thing at a time to find what breaks
    print("\n" + "=" * 60)
    print("PHASE 1b: ISOLATION — change ONE variable from baseline")
    print("=" * 60)

    # hange ONLY inChannels 2->3
    generate_conv2d_test(
        os.path.join(OUTPUT_DIR, "conv_p0_iso_3inch"),
        inputH=3, inputW=3, inChannels=3, outChannels=2,
        kH=2, kW=2, stride=1, padding=0, precision=0,
        label="ISO-A: only inCh 2->3")

    # Change ONLY outChannels 2->4
    generate_conv2d_test(
        os.path.join(OUTPUT_DIR, "conv_p0_iso_4outch"),
        inputH=3, inputW=3, inChannels=2, outChannels=4,
        kH=2, kW=2, stride=1, padding=0, precision=0,
        label="ISO-B: only outCh 2->4")

    # Change ONLY filter size 2x2->3x3
    generate_conv2d_test(
        os.path.join(OUTPUT_DIR, "conv_p0_iso_3x3filt"),
        inputH=3, inputW=3, inChannels=2, outChannels=2,
        kH=3, kW=3, stride=1, padding=0, precision=0,
        label="ISO-C: only filter 2x2->3x3")

    # Change ONLY padding 0->1
    generate_conv2d_test(
        os.path.join(OUTPUT_DIR, "conv_p0_iso_pad1"),
        inputH=3, inputW=3, inChannels=2, outChannels=2,
        kH=2, kW=2, stride=1, padding=1, precision=0,
        label="ISO-D: only padding 0->1")

    # Change ONLY input size 3x3->8x8
    generate_conv2d_test(
        os.path.join(OUTPUT_DIR, "conv_p0_iso_8x8"),
        inputH=8, inputW=8, inChannels=2, outChannels=2,
        kH=2, kW=2, stride=1, padding=0, precision=0,
        label="ISO-E: only size 3x3->8x8")

    # Everything medium EXCEPT padding stays 0
    generate_conv2d_test(
        os.path.join(OUTPUT_DIR, "conv_p0_medium_nopad"),
        inputH=8, inputW=8, inChannels=3, outChannels=4,
        kH=3, kW=3, stride=1, padding=0, precision=0,
        label="ISO-F: medium all EXCEPT pad=0")

    # precision=0, medium (known to FAIL)
    print("\n" + "=" * 60)
    print("PHASE 2: precision=0, medium (known to FAIL)")
    print("  8x8 in, 3ch->4ch, 3x3 filt, stride=1, pad=1")
    print("=" * 60)

    generate_fc_test(
        os.path.join(OUTPUT_DIR, "fc_p0_medium"),
        inputDim=16, outputDim=8, precision=0,
        label="FC p0 16x8")

    generate_conv2d_test(
        os.path.join(OUTPUT_DIR, "conv_p0_medium"),
        inputH=8, inputW=8, inChannels=3, outChannels=4,
        kH=3, kW=3, stride=1, padding=1, precision=0,
        label="MEDIUM: 8x8 3ch->4ch 3x3filt pad=1")

    # precision=4, tiny
    print("\n" + "=" * 60)
    print("PHASE 3: precision=4, tiny")
    print("=" * 60)

    generate_fc_test(
        os.path.join(OUTPUT_DIR, "fc_p4_tiny"),
        inputDim=4, outputDim=3, precision=4,
        label="FC p4 4x3")

    generate_conv2d_test(
        os.path.join(OUTPUT_DIR, "conv_p4_tiny"),
        inputH=3, inputW=3, inChannels=2, outChannels=2,
        kH=2, kW=2, stride=1, padding=0, precision=4,
        label="Conv2D p4 3x3in 2x2filt 2ch")

    # precision=8, medium
    print("\n" + "=" * 60)
    print("PHASE 4: precision=8, medium")
    print("=" * 60)

    generate_fc_test(
        os.path.join(OUTPUT_DIR, "fc_p8_medium"),
        inputDim=16, outputDim=8, precision=8,
        label="FC p8 16x8")

    generate_conv2d_test(
        os.path.join(OUTPUT_DIR, "conv_p8_medium"),
        inputH=8, inputW=8, inChannels=3, outChannels=4,
        kH=3, kW=3, stride=1, padding=1, precision=8,
        label="Conv2D p8 8x8in 3x3filt 3ch->4ch")

    # precision=12, medium
    print("\n" + "=" * 60)
    print("PHASE 5: precision=12, medium")
    print("=" * 60)

    generate_fc_test(
        os.path.join(OUTPUT_DIR, "fc_p12_medium"),
        inputDim=16, outputDim=8, precision=12,
        label="FC p12 16x8")

    generate_conv2d_test(
        os.path.join(OUTPUT_DIR, "conv_p12_medium"),
        inputH=8, inputW=8, inChannels=3, outChannels=4,
        kH=3, kW=3, stride=1, padding=1, precision=12,
        label="Conv2D p12 8x8in 3x3filt 3ch->4ch")

    print(f"\n{'=' * 60}")
    print(f"Done! Test data saved to {OUTPUT_DIR}/")
    for d in sorted(os.listdir(OUTPUT_DIR)):
        full = os.path.join(OUTPUT_DIR, d)
        if os.path.isdir(full):
            count = len([f for f in os.listdir(full) if f.endswith('.bin')])
            print(f"  {d}/ ({count} .bin files)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()