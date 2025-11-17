import os
from neural_field import (
    train_neural_field,
    render_image,
    plot_psnr_curve,
)
from PIL import Image

def run_single(image_path, tag):
    """
    Runs training + rendering + psnr + snapshots for one image.
    tag = 'test1' or 'test2'
    """
    output_dir = f"outputs_part1_{tag}"
    os.makedirs(output_dir, exist_ok=True)

    snapshot_iters = [0, 100, 250, 500, 1000, 2000, 3000]

    model, dataloader, history = train_neural_field(
        image_path=image_path,
        max_freq_log2=10,
        hidden_dim=256,
        num_hidden_layers=6,
        lr=1e-3,
        num_iterations=3000,
        batch_size=10000,
        eval_interval=100,
        device="cpu",
        snapshot_iters=snapshot_iters,
        snapshot_dir=os.path.join(output_dir, "snapshots"),
    )

    # Reconstruction output
    recon_img = render_image(model, dataloader, device="cpu")
    recon_path = os.path.join(output_dir, f"reconstruction_{tag}.png")
    recon_img.save(recon_path)
    print(f"[{tag}] Saved reconstruction → {recon_path}")

    # PSNR curve
    psnr_path = os.path.join(output_dir, f"psnr_{tag}.png")
    plot_psnr_curve(history, psnr_path)
    print(f"[{tag}] Saved PSNR curve → {psnr_path}")

    # Snapshot progression row
    iters_for_row = [0, 100, 250, 500, 1000, 3000]
    imgs = [history["snapshots"][i] for i in iters_for_row]
    widths, heights = zip(*(im.size for im in imgs))
    total_width = sum(widths)
    max_h = max(heights)

    row = Image.new("RGB", (total_width, max_h))
    x_offset = 0
    for im in imgs:
        row.paste(im, (x_offset, 0))
        x_offset += im.size[0]

    row_path = os.path.join(output_dir, f"progression_row_{tag}.png")
    row.save(row_path)
    print(f"[{tag}] Saved progression row → {row_path}")


def main():
    images = [
       # ("test1.jpg", "test1"),
        ("test2.jpg", "test2"),
    ]

    for img_path, tag in images:
        print(f"\n===== Running on {img_path} =====")
        run_single(img_path, tag)


if __name__ == "__main__":
    main()
