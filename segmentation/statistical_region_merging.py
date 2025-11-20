import os
import argparse
import numpy as np
from skimage import io, color, img_as_float
from skimage.filters import gaussian


def srm_segment(image, Q=32, gaussian_sigma=1.0):
    """Segment an RGB or grayscale image with Statistical Region Merging (SRM).

    Parameters
    ----------
    image : (H, W, 3) or (H, W) ndarray
        Input image. Can be uint8 or float; will be converted internally.
    Q : float
        Scale parameter of SRM. Smaller = fewer, larger regions.
    gaussian_sigma : float
        Sigma of the Gaussian pre-smoothing (in pixels).

    Returns
    -------
    labels : (H, W) ndarray of int
        Label map (0..K-1) of the final regions.
    seg_image : (H, W, 3) ndarray of float
        Segmented image where each region is filled with its mean intensity.
        (grayscale repeated on 3 channels for convenience).
    """
    # Convert to float grayscale in [0, 1]
    if image.ndim == 3:
        # skimage color.rgb2gray expects float in [0, 1]
        img = img_as_float(image)
        gray = color.rgb2gray(img)
    else:
        gray = img_as_float(image)

    # Optional smoothing (as in many SRM implementations)
    if gaussian_sigma is not None and gaussian_sigma > 0:
        gray = gaussian(gray, sigma=gaussian_sigma, mode="reflect")

    H, W = gray.shape
    n_pixels = H * W

    # Flatten grayscale image to 1D
    values = gray.reshape(-1)

    # Initialize Union-Find (Disjoint Set Union) structures
    parent = np.arange(n_pixels, dtype=np.int32)
    rank = np.zeros(n_pixels, dtype=np.int16)
    size = np.ones(n_pixels, dtype=np.int32)
    mean = values.copy()

    # Parameters for the merging predicate
    g = 1.0  # gray values are in [0, 1]
    logdelta = 2.0 * np.log(6.0 * n_pixels)

    def find_root(x):
        """Find with path compression."""
        root = x
        while parent[root] != root:
            root = parent[root]
        # Path compression
        while parent[x] != x:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root

    def threshold(sz):
        """Threshold b(R) (slightly simplified from Nock & Nielsen)."""
        return (g * g / (2.0 * Q * sz)) * (np.log(1.0 + sz) + logdelta)

    # Build 4-connected neighbor list (pairs of pixel indices)
    pairs = []
    for y in range(H):
        for x in range(W):
            idx = y * W + x
            if x + 1 < W:  # right neighbor
                pairs.append((idx, idx + 1))
            if y + 1 < H:  # bottom neighbor
                pairs.append((idx, idx + W))

    pairs = np.array(pairs, dtype=np.int32)

    # Sort pairs by absolute difference of gray levels (SRM ordering)
    diff = np.abs(values[pairs[:, 0]] - values[pairs[:, 1]])
    order = np.argsort(diff)
    pairs = pairs[order]

    # Main SRM merging loop
    for p, q in pairs:
        rp = find_root(p)
        rq = find_root(q)
        if rp == rq:
            continue

        dR = (mean[rp] - mean[rq]) ** 2
        dev = threshold(size[rp]) + threshold(size[rq])

        if dR < dev:
            # Union by rank
            if rank[rp] > rank[rq]:
                parent[rq] = rp
                new_root = rp
                other = rq
            elif rank[rp] < rank[rq]:
                parent[rp] = rq
                new_root = rq
                other = rp
            else:
                parent[rq] = rp
                rank[rp] += 1
                new_root = rp
                other = rq

            # Update stats for the merged region
            new_size = size[rp] + size[rq]
            new_mean = (mean[rp] * size[rp] + mean[rq] * size[rq]) / new_size
            size[new_root] = new_size
            mean[new_root] = new_mean
            size[other] = new_size
            mean[other] = new_mean

    # Second pass: assign compact labels 0..K-1
    roots = np.array([find_root(i) for i in range(n_pixels)], dtype=np.int32)
    unique_roots, labels_flat = np.unique(roots, return_inverse=True)
    labels = labels_flat.reshape(H, W)

    # Build segmented image: fill each region with its mean gray value
    region_means = mean[unique_roots]
    seg_gray = region_means[labels]
    seg_image = np.stack([seg_gray] * 3, axis=-1)  # fake RGB

    return labels, seg_image


def process_folder(input_folder, output_folder, Q=32, gaussian_sigma=1.0):
    os.makedirs(output_folder, exist_ok=True)
    # Common image extensions
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

    for fname in sorted(os.listdir(input_folder)):
        root, ext = os.path.splitext(fname)
        if ext.lower() not in exts:
            continue

        in_path = os.path.join(input_folder, fname)
        image = io.imread(in_path)

        labels, seg_image = srm_segment(image, Q=Q, gaussian_sigma=gaussian_sigma)

        # Save segmented image (uint8)
        out_img_path = os.path.join(output_folder, root + "_srm.png")
        io.imsave(out_img_path, (seg_image * 255).astype(np.uint8))

        # Optionally, save the label map as .npy
        out_lbl_path = os.path.join(output_folder, root + "_srm_labels.npy")
        np.save(out_lbl_path, labels)
        print(f"Processed {fname} -> {out_img_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Statistical Region Merging (SRM) segmentation for all images in a folder."
    )
    parser.add_argument(
        "--input", "-i", type=str, default="dataset",
        help="Input folder with images."
    )
    parser.add_argument(
        "--output", "-o", type=str, default="output_srm",
        help="Output folder for SRM results."
    )
    parser.add_argument(
        "--Q", type=float, default=32.0,
        help="SRM scale parameter Q (smaller = fewer regions)."
    )
    parser.add_argument(
        "--sigma", type=float, default=1.0,
        help="Gaussian smoothing sigma (0 to disable)."
    )
    args = parser.parse_args()

    process_folder(args.input, args.output, Q=args.Q, gaussian_sigma=args.sigma)


if __name__ == "__main__":
    main()
