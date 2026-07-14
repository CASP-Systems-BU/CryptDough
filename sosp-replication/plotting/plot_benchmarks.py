import csv
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os
from matplotlib.patches import Patch

DATA_DIR = "data"
OUT_DIR  = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

BASE_FONT  = 28
ANNOT_FONT = 22
TICK_FONT  = 28

plt.rcParams.update({
    "font.family": "serif",
    "font.size": BASE_FONT,
    "axes.labelsize": BASE_FONT,
    "xtick.labelsize": TICK_FONT,
    "ytick.labelsize": TICK_FONT,
    "legend.fontsize": TICK_FONT,
    "figure.dpi": 180,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PROTOCOL_COLOR = {
    "2PC": "#B9770E", # golden brown
    "3PC": "#1B7837", # forest green
    "4PC": "#C0392B", # brick red
    "spdz2k": "#27628e"  # blue-gray
}

COMP_COLOR = "#CCCCCC"

COMP_STYLE = {
    "orq": {"color": COMP_COLOR, "hatch": "////", "label": "ORQ"},
    "tva": {"color": COMP_COLOR, "hatch": "xxxx", "label": "TVA"},
    "pigeon": {"color": COMP_COLOR, "hatch": "....", "label": "Pigeon"},
    "piranha": {"color": COMP_COLOR, "hatch": "----", "label": "Piranha"},
    "mpspdz": {"color": COMP_COLOR, "hatch": "||||", "label": "MP-SPDZ"},
}

BAR_W = 0.25
SENTINEL = 1e-2 # placeholder for "none" values

def _to_float(s):
    """Return float or None for blank / error cells."""
    s = s.strip().replace(",", "")
    if not s or s.startswith("#"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def load_csv(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    if not rows:
        return {}

    # drop empty rows at the top
    while rows and all(c.strip() == "" for c in rows[0]):
        rows.pop(0)

    if rows and not any(c.strip() for c in rows[0][1:]):
        rows.pop(0)

    if len(rows) < 3:
        return {}

    protocol_row = rows[0]
    network_row  = rows[1]
    data_rows = rows[3:]

    blocks = [] # list of (protocol, network, start_col)
    cur_protocol = ""
    cur_network  = ""
    col = 1
    while col < len(protocol_row):
        p = protocol_row[col].strip()
        n = network_row[col].strip()
        if p:
            cur_protocol = p
        if n:
            cur_network = n
        if cur_protocol and cur_network:
            blocks.append((cur_protocol, cur_network, col))
            cur_network = "" # consume; next non-empty network cell starts new block
        col += 3 # each block is exactly 3 columns wide

    # Build output dict
    result = {}
    for protocol, network, start in blocks:
        queries = []
        competitors = []
        us_vals = []
        speedups = []
        for row in data_rows:
            if not row or not row[0].strip():
                continue
            queries.append(row[0].strip())
            competitors.append(_to_float(row[start]) if start < len(row) else None)
            us_vals.append(_to_float(row[start + 1]) if start + 1 < len(row) else None)
            speedups.append(_to_float(row[start + 2]) if start + 2 < len(row) else None)

        result.setdefault(protocol, {})[network] = {
            "query": queries,
            "competitor": competitors,
            "us": us_vals,
            "speedup": speedups,
        }

    return result

def load_multi_workload(filename):
    path = os.path.join(DATA_DIR, filename)

    protocols = []
    lan = []
    wan = []

    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            protocols.append(r["Protocol"])
            lan.append(_to_float(r["LAN"]))
            wan.append(_to_float(r["WAN"]))

    return protocols, lan, wan

def secrecy_bar_plot(ax, x_labels, competitor_vals, our_vals, speedups,
                     competitor_key, protocol="2PC",
                     log_scale=True, bar_width=BAR_W, rotate_xticks=30,
                     show_ylabel=True, show_legend=True):
    n  = len(x_labels)
    xs = np.arange(n)

    def safe(v):
        return [x if x is not None else SENTINEL for x in v]

    comp_s = safe(competitor_vals)
    ours_s = safe(our_vals)
    c_style = COMP_STYLE[competitor_key]
    our_color = PROTOCOL_COLOR[protocol]

    ax.bar(xs - bar_width / 2, comp_s,
           width=bar_width, color=c_style["color"], hatch=c_style["hatch"],
           edgecolor="black", linewidth=0.6, label=c_style["label"])

    ax.bar(xs + bar_width / 2, ours_s, width=bar_width, color=our_color, hatch="", edgecolor="black", linewidth=0.6, label="SysX")

    # y-axis
    real_vals = [v for v in comp_s + ours_s if v > SENTINEL * 2]

    if log_scale:
        ax.set_yscale("log")

        if real_vals:
            exp_lo = int(math.floor(math.log10(min(real_vals)))) - 1
            exp_hi = int(math.ceil(math.log10(max(real_vals))))
        else:
            exp_lo, exp_hi = -2, 4

        decade_ticks = [10.0 ** e for e in range(exp_lo, exp_hi + 1)]

        def fmt_decade(x):
            exp = int(round(math.log10(x)))
            return r"$10^{" + str(exp) + r"}$"

        ax.yaxis.set_major_locator(ticker.FixedLocator(decade_ticks))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: fmt_decade(x)))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.yaxis.grid(True, which="major", linestyle="-", linewidth=0.65, color="0.75", zorder=0)
        ax.yaxis.grid(True, which="minor", linestyle=":", linewidth=0.3, color="0.88", zorder=0)
        ax.set_ylim(bottom=10.0 ** exp_lo, top=10.0 ** exp_hi)
    else:
        ax.yaxis.grid(True, which="major", linestyle="-", linewidth=0.5, color="0.75", zorder=0)

    ax.xaxis.grid(True, linestyle="--", linewidth=0.35, color="0.82", zorder=0)
    ax.set_axisbelow(True)

    # speedup annotations
    for i, (cv, ov, su) in enumerate(zip(comp_s, ours_s, speedups)):
        if su is None:
            continue
        top  = max(cv, ov)
        ytop = top * 1.15
        ax.text(xs[i], ytop, f"{su:.1f}x", ha="center", va="bottom", fontsize=ANNOT_FONT, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, rotation=rotate_xticks, ha="right" if rotate_xticks > 0 else "center")
    if show_ylabel:
        ax.set_ylabel("Time (s)")
    if show_legend:
        ax.legend(loc="upper left", frameon=False, ncol=2)

## for plotting bars without a competitor
def ours_bar_plot(ax, x_labels, vals, protocol, log_scale=False, rotate_xticks=20):
    xs = np.arange(len(x_labels))
    vals_s = [v if v is not None else SENTINEL for v in vals]

    ax.bar(xs, vals_s,
           width=0.55,
           color=PROTOCOL_COLOR.get(protocol, "#333333"),
           edgecolor="black",
           linewidth=0.6)

    if log_scale:
        ax.set_yscale("log")

    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, rotation=rotate_xticks, ha="right")
    ax.set_ylabel("Time (s)")
    ax.yaxis.grid(True, linestyle="-", linewidth=0.5, color="0.75")
    ax.xaxis.grid(True, linestyle="--", linewidth=0.35, color="0.82")
    ax.set_axisbelow(True)


def save(fig, name):
    path = os.path.join(OUT_DIR, name.replace(".pdf", ".png"))
    fig.savefig(path, bbox_inches="tight", dpi=180)
    print(f"  saved {path}")
    plt.close(fig)

# per engine plot funcs
# x-axis label overrides
LABEL_OVERRIDES = {
    "Credit Score": "Credit\nScore",
    "Market Share": "Market\nShare",
    "AlexNet + CIFAR-10": "AlexNet\nCIFAR-10",
    "VGG16 + CIFAR-10": "VGG16\nCIFAR-10",
    "VGG16 + ImageNet": "VGG16\nImageNet",
}

def apply_label_overrides(labels):
    return [LABEL_OVERRIDES.get(l, l) for l in labels]

def plot_orq():
    data = load_csv("ORQ.csv")
    for protocol in ["2PC", "3PC"]:
        for network in ["LAN", "WAN"]:
            d = data[protocol][network]
            fig, ax = plt.subplots(figsize=(10, 8))
            secrecy_bar_plot(ax,
                             x_labels=apply_label_overrides(d["query"]),
                             competitor_vals=d["competitor"],
                             our_vals=d["us"],
                             speedups=d["speedup"],
                             competitor_key="orq",
                             protocol=protocol,
                             show_ylabel=(network == "LAN" and protocol == "2PC"),
                             show_legend=(network == "LAN"))
            fig.tight_layout()
            save(fig, f"orq_{protocol}_{network}.pdf")


def plot_tva():
    data = load_csv("TVA.csv")
    for protocol in ["3PC", "4PC"]:
        for network in ["LAN", "WAN"]:
            d = data[protocol][network]
            fig, ax = plt.subplots(figsize=(10, 8))
            secrecy_bar_plot(ax,
                             x_labels=apply_label_overrides(d["query"]),
                             competitor_vals=d["competitor"],
                             our_vals=d["us"],
                             speedups=d["speedup"],
                             competitor_key="tva",
                             protocol=protocol,
                             show_ylabel=(network == "LAN" and protocol == "3PC"),
                             show_legend=(network == "WAN"))
            fig.tight_layout()
            save(fig, f"tva_{protocol}_{network}.pdf")


def plot_pigeon():
    data = load_csv("Pigeon.csv")
    for protocol in ["3PC", "4PC"]:
        for network in ["LAN", "WAN"]:
            d = data[protocol][network]
            fig, ax = plt.subplots(figsize=(10, 8))
            secrecy_bar_plot(ax,
                             x_labels=apply_label_overrides(d["query"]),
                             competitor_vals=d["competitor"],
                             our_vals=d["us"],
                             speedups=d["speedup"],
                             competitor_key="pigeon",
                             protocol=protocol,
                             rotate_xticks=0,
                             show_ylabel=(network == "WAN"),
                             show_legend=(network == "WAN"))
            fig.tight_layout()
            save(fig, f"pigeon_{protocol}_{network}.pdf")


def plot_piranha():
    data = load_csv("Piranha.csv")
    for network in ["LAN", "WAN"]:
        d = data["2PC"][network]
        fig, ax = plt.subplots(figsize=(10, 8))
        secrecy_bar_plot(ax,
                         x_labels=apply_label_overrides(d["query"]),
                         competitor_vals=d["competitor"],
                         our_vals=d["us"],
                         speedups=d["speedup"],
                         competitor_key="piranha",
                         protocol="2PC",
                         rotate_xticks=0,
                         show_ylabel=(network == "LAN"),
                         show_legend=(network == "WAN"))
        fig.tight_layout()
        save(fig, f"piranha_2PC_{network}.pdf")


def plot_mpspdz():
    groups    = ["AlexNet LAN", "AlexNet WAN"]
    data = load_csv("MP-SPDZ.csv")
    dlan = data["spdz2k"]["LAN"]
    dwan = data["spdz2k"]["WAN"]
    fig, ax = plt.subplots(figsize=(10, 8))
    secrecy_bar_plot(ax,
                     x_labels=groups,
                     competitor_vals=dlan["competitor"] + dwan["competitor"],
                     our_vals=dlan["us"] + dwan["us"],
                     speedups=dlan["speedup"] + dwan["speedup"],
                     competitor_key="mpspdz",
                     protocol="spdz2k",
                     rotate_xticks=0)
    fig.tight_layout()
    ax.legend(loc="upper left", frameon=False, ncol=1)
    save(fig, "mpspdz_alexnet_cifar10.pdf")

## threat model annotation
def add_group_bracket(ax, x0, x1, label, y, direction="up", color="black", h=0.03):

    if direction == "up":
        ys = [y, y + h, y + h, y]
        text_y = y + h + 0.02
        va = "bottom"
    else:
        ys = [y, y - h, y - h, y]
        text_y = y - h - 0.02
        va = "top"

    ax.plot([x0, x0, x1, x1],
            ys,
            transform=ax.get_xaxis_transform(),
            color=color,
            linewidth=1.2,
            clip_on=False)

    txt = ax.text((x0 + x1) / 2, text_y,
            label,
            ha="center",
            va=va,
            transform=ax.get_xaxis_transform(),
            fontstyle="italic",
            fontsize=ANNOT_FONT,
            color=color)
    
    if label == "Dishonest Majority":
        txt.set_fontweight("bold")

## multi-worklad query plots
def plot_multi_workload():
    protocols, lan_vals, wan_vals = load_multi_workload("MULTI_WORKLOAD.csv")

    # seconds → minutes
    lan_vals = [v/60.0 if v is not None else None for v in lan_vals]
    wan_vals = [v/60.0 if v is not None else None for v in wan_vals]

    fig, ax = plt.subplots(figsize=(14, 7))

    xs = np.arange(len(protocols))
    width = 0.35

    colors = [PROTOCOL_COLOR.get(p, "#333333") for p in protocols]

    ax.bar(xs - width/2, lan_vals,
           width=width,
           color=colors,
           edgecolor="black",
           linewidth=1.0,
           alpha=0.9,
           label="LAN")

    ax.bar(xs + width/2, wan_vals,
           width=width,
           color=colors,
           edgecolor="black",
           linewidth=1.0,
           hatch="//",
           alpha=0.9,
           label="WAN")

    ax.set_xticks(xs)
    ax.set_xticklabels(["ABY", "ABY3", "F4", "SPDZ2k"])

    DISHONEST_COLOR = "#7A8694"
    for label, proto in zip(ax.get_xticklabels(), protocols):
        if proto in ["2PC", "spdz2k"]:
            #label.set_color(DISHONEST_COLOR)
            label.set_fontweight("bold")

    # threat model annotations
    add_group_bracket(ax, -0.2, 1.2, "Semi-honest", y=0.40, direction="up")
    add_group_bracket(ax, 1.8, 3.2, "Malicious", y=1, direction="up")

    add_group_bracket(ax, 0.8, 2.2, "Honest Majority", y=-0.15, direction="down")
    add_group_bracket(ax, -0.2, 3.2, "Dishonest Majority", y=-0.2, direction="down", h=0.1)#, color=DISHONEST_COLOR)

    ax.set_ylabel("Time (min)")
    legend_handles = [
    Patch(facecolor="white", edgecolor="black", label="LAN"),
    Patch(facecolor="white", edgecolor="black", hatch="//", label="WAN"),
    ]
    ax.legend(handles=legend_handles, frameon=False)

    ax.yaxis.grid(True, linestyle="-", linewidth=0.6, color="0.85")
    ax.xaxis.grid(False)
    ax.set_yscale('log')
    ax.set_axisbelow(True)

    fig.subplots_adjust(bottom=0.35, top=0.82)
    fig.tight_layout()
    save(fig, "multi_workload.pdf")

def scalability_line_plot(ax, x, y, label, color,
                          marker="o", linestyle="-"):
    
    y = [v / 60.0 for v in y] # minutes
    ax.plot(x, y,
            marker=marker,
            linestyle=linestyle,
            linewidth=3,
            markersize=9,
            color=color,
            label=label)

    ax.set_xlabel("#threads")
    ax.set_ylabel("Time (min)")
    ax.grid(True, linestyle="-", linewidth=0.5, color="0.75")

def plot_sorting_scalability():
    path = os.path.join(DATA_DIR, "SORTING_SCALABILITY.csv")

    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    protocol = "3PC"
    networks = ["LAN"]

    for network in networks:

        per_query = {}

        for r in rows:
            if r["Network"] != network:
                continue

            q = r["Query"]
            t = int(r["Threads"])
            v = float(r["Ours"])

            per_query.setdefault(q, []).append((t, v))

        fig, ax = plt.subplots(figsize=(10, 6))

        markers = ["o", "x"]

        for i, (q, pts) in enumerate(per_query.items()):
            pts.sort()
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]

            scalability_line_plot(
                ax,
                xs,
                ys,
                label=q,
                color=PROTOCOL_COLOR[protocol],
                marker=markers[i]
            )

        ax.set_xticks(sorted({t for pts in per_query.values() for (t, _) in pts}))
        ax.legend(frameon=False)

        fig.tight_layout()
        save(fig, f"sorting_scalability_{protocol}_{network}.pdf")

def scalability_line_plot_sec(ax, x, y, label, color, marker="o", linestyle="-"):
    ax.plot(x, y,
            marker=marker,
            linestyle=linestyle,
            linewidth=3,
            markersize=9,
            color=color,
            label=label)

    ax.set_xlabel("#threads")
    ax.set_ylabel("Time (s)")
    ax.grid(True, linestyle="-", linewidth=0.5, color="0.75")

def plot_gr_scalability():
    path = os.path.join(DATA_DIR, "GR_SCALABILITY.csv")

    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    protocol = "3PC"

    pts = []
    for r in rows:
        if r["Network"] != "LAN":
            continue
        pts.append((int(r["Threads"]), float(r["Ours"])))

    pts.sort()
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    fig, ax = plt.subplots(figsize=(10, 6))
    scalability_line_plot_sec(ax, xs, ys, label="comparison", color="#27628e", marker="o")
    ax.set_xticks(sorted(set(xs)))
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "gr_scalability_3PC_LAN.pdf")


def plot_rca_ppa_scalability():
    path = os.path.join(DATA_DIR, "RCA-PPA_SCALABILITY.csv")

    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    protocol = "3PC"
    per_query = {}

    for r in rows:
        if r["Network"] != "LAN":
            continue
        q = r["Query"]
        t = int(r["Threads"])
        v = float(r["Ours"])
        per_query.setdefault(q, []).append((t, v))

    fig, ax = plt.subplots(figsize=(10, 6))
    markers = ["o", "x"]

    for i, (q, pts) in enumerate(per_query.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        scalability_line_plot_sec(ax, xs, ys, label=q, color="#B9770E", marker=markers[i])

    ax.set_xticks(sorted({t for pts in per_query.values() for (t, _) in pts}))
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "rca_ppa_scalability_3PC_LAN.pdf")


def plot_conv2d_scalability():
    path = os.path.join(DATA_DIR, "CONV2D_SCALABILITY.csv")

    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    protocol = "3PC"
    batch_cols = ["batch -12"]
    markers = ["o"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, col in enumerate(batch_cols):
        xs = [int(r["Threads"]) for r in rows]
        ys = [float(r[col]) for r in rows]
        scalability_line_plot_sec(ax, xs, ys, label="conv2D", color="#C0392B", marker=markers[i])

    ax.set_xticks(sorted(set(int(r["Threads"]) for r in rows)))
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "conv2d_scalability_3PC_LAN.pdf")


def load_scalability_sheet(filename):
    import re
    path = os.path.join(DATA_DIR, filename)
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    COL_TO_PROTO = {"3pc": "3PC", "2pc": "2PC", "4pc": "4PC", "spdz": "spdz2k"}
    data = {}
    current_section = None
    col_map = {}

    for row in rows:
        if not row or not any(c.strip() for c in row):
            continue
        first = row[0].strip()
        m = re.search(r"Primitives scaling - (\w+)", first, re.IGNORECASE)
        if m:
            current_section = m.group(1).lower()
            data.setdefault(current_section, {})
            col_map = {}
            continue
        if first.lower() == "threads" and current_section:
            col_map = {i: COL_TO_PROTO[h.strip().lower()]
                       for i, h in enumerate(row) if h.strip().lower() in COL_TO_PROTO}
            continue
        if current_section and col_map:
            try:
                threads = int(first)
            except ValueError:
                continue
            for col_idx, proto in col_map.items():
                val = _to_float(row[col_idx]) if col_idx < len(row) else None
                data[current_section].setdefault(proto, []).append((threads, val))

    for prim in data:
        for proto in data[prim]:
            data[prim][proto].sort(key=lambda t: t[0])
    return data


def plot_primitives_scalability():
    data = load_scalability_sheet("scalability_primitives.csv")

    PANELS = [
        ("a", "Comparison", ["gr"],          ["comparison"], ["o"]),
        ("b", "RCA/PPA",    ["rca", "ppa"],  ["RCA", "PPA"], ["o", "x"]),
        ("c", "Conv2D",     ["conv2d"],       ["conv2D"],     ["o"]),
        ("d", "Mul", ["mul"], ["Mul"], ["o"]),
    ]

    for proto in ["2PC", "3PC", "4PC", "spdz2k"]:
        color = PROTOCOL_COLOR[proto]

        for _letter, caption, primitives, line_labels, markers in PANELS:
            fig, ax = plt.subplots(figsize=(10, 6))
            all_threads = set()
            for prim, lbl, mrk in zip(primitives, line_labels, markers):
                pts = [(t, v) for t, v in data.get(prim, {}).get(proto, []) if v is not None]
                if not pts:
                    continue
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                all_threads.update(xs)
                scalability_line_plot_sec(ax, xs, ys, label=lbl, color=color, marker=mrk)
            if all_threads:
                ax.set_xticks(sorted(all_threads))
            ax.legend(frameon=False, loc="upper right")
            fig.tight_layout()
            fname = caption.replace("/", "_")
            save(fig, f"primitives_scalability_{proto}_{fname}_LAN.pdf")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate MPC benchmark plots")
    valid = ["orq", "tva", "pigeon", "piranha", "mpspdz", "multi", "sorting", "gr", "rca_ppa", "conv2d", "primitives"]
    parser.add_argument("engines", nargs="*", help=f"Which engines to plot (default: all). Choices: {valid}")
    args = parser.parse_args()

    engines = args.engines or valid
    for e in engines:
        if e not in valid:
            parser.error(f"invalid choice: {e!r} (choose from {valid})")
    dispatch = {
        "orq": plot_orq,
        "tva": plot_tva,
        "pigeon": plot_pigeon,
        "piranha": plot_piranha,
        "mpspdz": plot_mpspdz,
        "multi": plot_multi_workload,
        "sorting": plot_sorting_scalability,
        "gr": plot_gr_scalability,
        "rca_ppa": plot_rca_ppa_scalability,
        "conv2d": plot_conv2d_scalability,
        "primitives": plot_primitives_scalability,
    }
    for engine in engines:
        dispatch[engine]()
    print(f"\nPlots written to ./{OUT_DIR}/")