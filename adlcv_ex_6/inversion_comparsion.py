import os
import re
import numpy as np
import torch
import lpips
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# folder containing images
folder = "output"

# regex patterns
ddmi_pattern = re.compile(r"ddim_reconstruction_(\d+)\.png")
nti_pattern = re.compile(r"nti_reconstruction_(\d+)\.png")
orig_pattern = re.compile(r"original_(\d+)\.png")

# collect files
ddmi_files = {}
nti_files = {}
orig_files = {}

for f in os.listdir(folder):
    if m := ddmi_pattern.match(f):
        ddmi_files[m.group(1)] = f
    elif m := nti_pattern.match(f):
        nti_files[m.group(1)] = f
    elif m := orig_pattern.match(f):
        orig_files[m.group(1)] = f

ids = sorted(set(orig_files.keys()) & set(ddmi_files.keys()) & set(nti_files.keys()))

# LPIPS model
lpips_model = lpips.LPIPS(net="alex")

# results
results = {
    "ddmi": {"psnr": [], "ssim": [], "lpips": []},
    "nti": {"psnr": [], "ssim": [], "lpips": []},
}

def load_image(path):
    img = Image.open(path).convert("RGB")
    return np.array(img) / 255.0

def to_lpips_tensor(img):
    t = torch.tensor(img).permute(2,0,1).unsqueeze(0).float()
    return t * 2 - 1

for id_ in tqdm(ids):

    orig = load_image(os.path.join(folder, orig_files[id_]))
    ddmi = load_image(os.path.join(folder, ddmi_files[id_]))
    nti = load_image(os.path.join(folder, nti_files[id_]))

    # --- PSNR ---
    results["ddmi"]["psnr"].append(peak_signal_noise_ratio(orig, ddmi, data_range=1))
    results["nti"]["psnr"].append(peak_signal_noise_ratio(orig, nti, data_range=1))

    # --- SSIM ---
    results["ddmi"]["ssim"].append(structural_similarity(orig, ddmi, channel_axis=2, data_range=1))
    results["nti"]["ssim"].append(structural_similarity(orig, nti, channel_axis=2, data_range=1))

    # --- LPIPS ---
    with torch.no_grad():
        lp_ddmi = lpips_model(to_lpips_tensor(orig), to_lpips_tensor(ddmi))
        lp_nti = lpips_model(to_lpips_tensor(orig), to_lpips_tensor(nti))

    results["ddmi"]["lpips"].append(lp_ddmi.item())
    results["nti"]["lpips"].append(lp_nti.item())

# --- averages ---
print("\nAverage Metrics\n")

for method in ["ddmi", "nti"]:
    print(method.upper())
    print("PSNR :", np.mean(results[method]["psnr"]))
    print("SSIM :", np.mean(results[method]["ssim"]))
    print("LPIPS:", np.mean(results[method]["lpips"]))
    print()

n_show = min(3, len(ids))

plt.figure(figsize=(10, 3 * n_show))

for i, id_ in enumerate(ids[:n_show]):
    if i == 0:
        pass
    orig = load_image(os.path.join(folder, orig_files[id_]))
    ddmi = load_image(os.path.join(folder, ddmi_files[id_]))
    nti = load_image(os.path.join(folder, nti_files[id_]))

    plt.subplot(n_show, 3, i * 3 + 1)
    plt.imshow(orig)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(n_show, 3, i * 3 + 2)
    plt.imshow(ddmi)
    plt.title("DDMI")
    plt.axis("off")

    plt.subplot(n_show, 3, i * 3 + 3)
    plt.imshow(nti)
    plt.title("NTI")
    plt.axis("off")

plt.tight_layout()
plt.show()

metrics = ["psnr", "ssim", "lpips"]

for m in metrics:
    plt.figure()
    plt.boxplot(
        [results["ddmi"][m], results["nti"][m]],
        labels=["DDMI", "NTI"]
    )
    plt.title(m.upper())
    plt.show()

def boxplot_stats(values):
    values = np.array(values)

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    median = np.median(values)
    mean = np.mean(values)

    iqr = q3 - q1

    lower_whisker = q1 - 1.5 * iqr
    upper_whisker = q3 + 1.5 * iqr

    non_outliers = values[(values >= lower_whisker) & (values <= upper_whisker)]
    whisker_low = non_outliers.min()
    whisker_high = non_outliers.max()

    outliers = values[(values < lower_whisker) | (values > upper_whisker)]

    return {
        "mean": mean,
        "median": median,
        "q1": q1,
        "q3": q3,
        "whisker_low": whisker_low,
        "whisker_high": whisker_high,
        "min": values.min(),
        "max": values.max(),
        "outliers": outliers
    }


metrics = ["psnr", "ssim", "lpips"]

for m in metrics:
    print("\n==============================")
    print(m.upper())
    print("==============================")

    for method in ["ddmi", "nti"]:

        stats = boxplot_stats(results[method][m])

        print(f"\n{method.upper()}")

        print(f"Mean: {stats['mean']:.4f}")
        print(f"Median: {stats['median']:.4f}")

        print(f"Q1: {stats['q1']:.4f}")
        print(f"Q3: {stats['q3']:.4f}")

        print(f"Whisker low: {stats['whisker_low']:.4f}")
        print(f"Whisker high: {stats['whisker_high']:.4f}")

        print(f"Min: {stats['min']:.4f}")
        print(f"Max: {stats['max']:.4f}")

        print(f"Number of outliers: {len(stats['outliers'])}")