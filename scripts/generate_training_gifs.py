# -*- coding: utf-8 -*-
"""
Created on 18th November 2026
Updated on 18th November 2026
@author: Lenovo Laptop

Generate crystal-clear, educational training animation GIFs for all 5 architectures
(NN, CNN, DBN, SAE, CAE) using the unified MNIST dataset and pure-NumPy models.

Each GIF visualizes:
- Internal layer activations and learned filters/receptive fields as they evolve
- Live layer-to-image reverse mappings back to 28x28 pixel space during training
- Live class predictions / reconstructions alongside training loss and accuracy curves
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

from data import mnist
from util import zscore, normalize, sigm, conv2d
from NN import NeuralNetwork, nntrain, nnff, nntest
from CNN import ConvolutionalNeuralNetwork, cnntrain, cnnff, cnntest
from DBN import DeepBeliefNetwork, dbntrain, dbnunfoldtonn
from SAE.StackedAutoEncoder import SAE
from SAE.saetrain import saetrain
from CAE.ConvolutionalAutoEncoder import CAE
from CAE.caeff import caeff
from CAE.caetrain import caetrain
from CAE.caetest import caetest


ASSETS_DIR = os.path.join(ROOT_DIR, "output")

# Clean high-contrast studio palette (works cleanly on GitHub light and dark themes)
BG_DARK = "#0d1117"
CARD_BG = "#161b22"
BORDER_CLR = "#30363d"
TEXT_MAIN = "#f0f6fc"
TEXT_MUTED = "#8b949e"
CYAN = "#58a6ff"
GREEN = "#3fb950"
CORAL = "#f78166"
AMBER = "#e3b341"
PURPLE = "#bc8cff"


def style_axes(ax, title=None):
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_color(BORDER_CLR)
        spine.set_linewidth(1.1)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8.5)
    if title:
        ax.set_title(title, color=TEXT_MAIN, fontsize=10, fontweight="bold", pad=6)


def hide_ticks(ax):
    ax.set_xticks([])
    ax.set_yticks([])


def fig_to_pil(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return Image.fromarray(buf[:, :, :3].copy(), mode="RGB")


def save_gif(frames, out_path, frame_ms=130, end_pause_ms=1600):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    quantized = [
        f.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        for f in frames
    ]
    durations = [frame_ms] * (len(quantized) - 1) + [end_pause_ms]
    quantized[0].save(
        out_path,
        save_all=True,
        append_images=quantized[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    kb = os.path.getsize(out_path) / 1024.0
    print(f"Saved {out_path} ({len(frames)} frames, {kb:.1f} KB)")
    if frames:
        png_path = os.path.splitext(out_path)[0] + "_final.png"
        frames[-1].save(png_path)
        print(f"Saved {png_path}")


def tile_images(imgs, rows, cols, pad=2, bg=0.0):
    """Arrange (N, H, W) images into a single (rows*H + pads, cols*W + pads) grid."""
    h, w = imgs.shape[1], imgs.shape[2]
    canvas = np.full(
        (rows * h + (rows - 1) * pad, cols * w + (cols - 1) * pad),
        bg,
        dtype=np.float64,
    )
    for idx in range(min(len(imgs), rows * cols)):
        r, c = divmod(idx, cols)
        y0 = r * (h + pad)
        x0 = c * (w + pad)
        canvas[y0:y0 + h, x0:x0 + w] = imgs[idx]
    return canvas


def load_unified_mnist():
    ds = mnist.load_mnist()
    train_x = ds["train_x"] / 255.0
    train_y = ds["train_y"]
    test_x = ds["test_x"] / 255.0
    test_y = ds["test_y"]
    return train_x, train_y, test_x, test_y


def select_probe_digits(test_x, test_y, digits=(0, 3, 5, 8)):
    """Pick clean representative MNIST test images for each requested digit."""
    labels = test_y.argmax(axis=1)
    idxs = []
    for d in digits:
        matches = np.where(labels == d)[0]
        # Pick the 2nd match if available for clean canonical strokes
        idxs.append(int(matches[min(1, len(matches) - 1)]))
    return idxs


def build_snapshot_steps(total_batches_per_epoch, num_epochs):
    """Front-loaded batch indices capturing early emergence and convergence."""
    early = [0, 2, 5, 10, 18, 30, 50, 80, 125, 180, 250, 350, 480]
    steps = set()
    for ep in range(num_epochs):
        if ep == 0:
            for b in early:
                if b < total_batches_per_epoch:
                    steps.add((ep, b))
        else:
            for frac in (0.3, 0.65):
                steps.add((ep, int(frac * (total_batches_per_epoch - 1))))
        steps.add((ep, total_batches_per_epoch - 1))
    return steps


# ---------------------------------------------------------------------------
# 1. NN Training Animation
# ---------------------------------------------------------------------------
def generate_nn_gif(train_x, train_y, test_x, test_y):
    print("\n--- Generating NN training GIF ---")
    tx_norm, mu, sigma = zscore.zscore(train_x)
    vx_norm = normalize.normalize(test_x, mu, sigma)

    probe_idx = select_probe_digits(test_x, test_y, digits=(3,))[0]
    probe_raw = test_x[probe_idx].reshape(28, 28)
    probe_x = vx_norm[probe_idx:probe_idx + 1]

    np.random.seed(1)
    nn = NeuralNetwork.NN(np.array([784, 100, 10]))
    nn.Output = "softmax"
    numepochs = 3
    batchsize = 100
    numbatches = tx_norm.shape[0] // batchsize
    snap_steps = build_snapshot_steps(numbatches, numepochs)

    frames = []
    hist_steps, hist_loss, hist_acc = [], [], []
    eval_sub_x, eval_sub_y = vx_norm[:2000], test_y[:2000]

    def render_frame(net, label_text):
        net.Testing = True
        net = nnff.nnff(net, probe_x, np.zeros((1, 10)))
        net.Testing = False
        hidden_act = net.A[1][0, 1:]
        probs = net.A[-1][0]
        pred = int(np.argmax(probs))

        # Live reverse map from target class '3' through hidden activations -> 28x28 pixels
        class_contrib = hidden_act * net.W[1][3, 1:]
        rev_map = (class_contrib @ net.W[0][:, 1:]).reshape(28, 28)
        rev_max = max(np.percentile(np.abs(rev_map), 99), 1e-3)

        # 16 hidden receptive fields (4x4 grid)
        rf = net.W[0][:16, 1:].reshape(16, 28, 28)
        vmax = max(np.percentile(np.abs(rf), 99), 1e-3)

        fig = plt.figure(figsize=(10.8, 5.4), dpi=100, facecolor=BG_DARK)
        gs = gridspec.GridSpec(
            2, 3, width_ratios=[1.15, 1.15, 1.25], height_ratios=[1.05, 1.0],
            wspace=0.28, hspace=0.40, left=0.04, right=0.97, top=0.83, bottom=0.11,
        )
        fig.suptitle(
            "Feedforward Neural Network (784 → 100 → 10, Tanh + Softmax)",
            color=TEXT_MAIN, fontsize=13, fontweight="bold", y=0.96,
        )
        acc_str = f"{hist_acc[-1]:.1f}%" if hist_acc else "--"
        loss_str = f"{hist_loss[-1]:.3f}" if hist_loss else "--"
        fig.text(
            0.5, 0.895,
            f"{label_text}   |   Batch Loss: {loss_str}   |   Test Accuracy: {acc_str}",
            ha="center", color=CYAN, fontsize=10, fontweight="bold",
        )

        # Left: Hidden receptive fields
        ax_rf = fig.add_subplot(gs[:, 0])
        style_axes(ax_rf, "Hidden Weights W[0] (16 of 100)")
        hide_ticks(ax_rf)
        ax_rf.imshow(tile_images(rf, 4, 4, pad=2, bg=0.0), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax_rf.set_xlabel("Red = +stroke template, Blue = -inhibitory", color=TEXT_MUTED, fontsize=8.5)

        # Middle top: Probe digit vs Live Reverse Map
        sub_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 1], wspace=0.12)
        ax_in = fig.add_subplot(sub_gs[0])
        style_axes(ax_in, "Probe Input '3'")
        hide_ticks(ax_in)
        ax_in.imshow(probe_raw, cmap="gray", vmin=0, vmax=1)

        ax_rev = fig.add_subplot(sub_gs[1])
        style_axes(ax_rev, "Live Reverse Map")
        hide_ticks(ax_rev)
        ax_rev.imshow(rev_map, cmap="RdBu_r", vmin=-rev_max, vmax=rev_max)
        ax_rev.set_xlabel("Class '3' → Hidden → Pixels", color=TEXT_MUTED, fontsize=8)

        # Middle bottom: Output probability bars
        ax_prob = fig.add_subplot(gs[1, 1])
        style_axes(ax_prob, f"Output Probabilities (Pred: {pred})")
        colors = [GREEN if k == 3 and pred == 3 else (CORAL if k == pred else CYAN) for k in range(10)]
        ax_prob.bar(np.arange(10), probs, color=colors, width=0.72)
        ax_prob.set_xticks(range(10))
        ax_prob.set_ylim(0, 1.08)
        ax_prob.set_ylabel("Softmax P(y)", color=TEXT_MUTED, fontsize=8.5)
        ax_prob.grid(axis="y", color=BORDER_CLR, alpha=0.4, linewidth=0.6)

        # Right top: Training loss
        ax_loss = fig.add_subplot(gs[0, 2])
        style_axes(ax_loss, "Cross-Entropy Training Loss")
        if hist_steps:
            ax_loss.plot(hist_steps, hist_loss, color=CORAL, linewidth=2.2)
            ax_loss.scatter([hist_steps[-1]], [hist_loss[-1]], color=TEXT_MAIN, s=28, zorder=5)
        ax_loss.set_xlim(0, numepochs * numbatches)
        ax_loss.set_ylim(0, 2.6)
        ax_loss.grid(color=BORDER_CLR, alpha=0.45, linewidth=0.6)

        # Right bottom: Test accuracy
        ax_acc = fig.add_subplot(gs[1, 2])
        style_axes(ax_acc, "Test Accuracy (%)")
        if hist_steps:
            ax_acc.plot(hist_steps, hist_acc, color=GREEN, linewidth=2.2)
            ax_acc.scatter([hist_steps[-1]], [hist_acc[-1]], color=TEXT_MAIN, s=28, zorder=5)
        ax_acc.set_xlim(0, numepochs * numbatches)
        ax_acc.set_ylim(0, 102)
        ax_acc.set_xlabel("Mini-batch Step", color=TEXT_MUTED, fontsize=8.5)
        ax_acc.grid(color=BORDER_CLR, alpha=0.45, linewidth=0.6)

        frames.append(fig_to_pil(fig))
        plt.close(fig)

    er0, _ = nntest.nntest(nn, eval_sub_x, eval_sub_y)
    hist_steps.append(0)
    hist_loss.append(2.30)
    hist_acc.append((1.0 - er0) * 100.0)
    render_frame(nn, "Epoch 0 (Random Initialization)")

    def on_step(net, ep, batch, loss_val, global_step):
        if (ep, batch) in snap_steps:
            er, _ = nntest.nntest(net, eval_sub_x, eval_sub_y)
            hist_steps.append(global_step + 1)
            hist_loss.append(float(loss_val))
            hist_acc.append((1.0 - er) * 100.0)
            render_frame(net, f"Epoch {ep + 1}/{numepochs}  •  Batch {batch + 1}/{numbatches}")

    nntrain.nntrain(nn, tx_norm, train_y, {"numepochs": numepochs, "batchsize": batchsize, "plot": 0},
                    None, None, None, callback=on_step)
    save_gif(frames, os.path.join(ASSETS_DIR, "nn_training.gif"))


# ---------------------------------------------------------------------------
# 2. CNN Training Animation (with Live Deconv Reverse Map to Image Space)
# ---------------------------------------------------------------------------
def cnn_reverse_map(net, probe_img):
    """
    Reverse-project active Conv2 feature maps back through Pool1 -> Conv1^T
    into 28x28 pixel space so viewers can see what Conv2 detects inside the digit.
    """
    net = cnnff.cnnff(net, probe_img)
    # Use Conv2 activations (8x8x1 across 12 maps) centered around mean activity
    c2_maps = {j: np.maximum(0.0, net.layers[3].A[j] - 0.5) for j in range(12)}
    k2_swap = {(j, i): net.layers[3].K[i, j] for i in range(6) for j in range(12)}
    p1_proj = conv2d.conv2d_full_batch(c2_maps, k2_swap, 6, rot180=True)
    c1_proj = {
        i: np.repeat(np.repeat(p1_proj[i], 2, axis=0), 2, axis=1) * net.layers[1].A[i] * (1.0 - net.layers[1].A[i])
        for i in range(6)
    }
    k1_swap = {(j, 0): net.layers[1].K[0, j] for j in range(6)}
    img_proj = conv2d.conv2d_full_batch(c1_proj, k1_swap, 1, rot180=True)[0][:, :, 0]
    return net, img_proj


def generate_cnn_gif(train_x, train_y, test_x, test_y):
    print("\n--- Generating CNN training GIF ---")
    cx_train = np.transpose(train_x.reshape(-1, 28, 28), (1, 2, 0))
    cy_train = train_y.T
    cx_test = np.transpose(test_x.reshape(-1, 28, 28), (1, 2, 0))
    cy_test = test_y.T

    probe_idx = select_probe_digits(test_x, test_y, digits=(4,))[0]
    probe_img = cx_test[:, :, probe_idx:probe_idx + 1]

    np.random.seed(1)
    cnn = ConvolutionalNeuralNetwork.CNN(cx_train, cy_train)
    numepochs = 2
    batchsize = 50
    numbatches = cx_train.shape[2] // batchsize
    snap_steps = build_snapshot_steps(numbatches, numepochs)

    frames = []
    hist_steps, hist_loss, hist_acc = [], [], []
    eval_x, eval_y = cx_test[:, :, :1000], cy_test[:, :1000]

    def render_frame(net, label_text):
        net, rev_img = cnn_reverse_map(net, probe_img)
        rev_max = max(np.percentile(np.abs(rev_img), 99), 1e-4)

        c1_maps = np.array([net.layers[1].A[j][:, :, 0] for j in range(6)])
        c2_maps = np.array([net.layers[3].A[j][:, :, 0] for j in range(12)])
        kernels = np.array([net.layers[1].K[0, j] for j in range(6)])
        k_max = max(np.max(np.abs(kernels)), 1e-3)
        probs = net.O[:, 0]
        pred = int(np.argmax(probs))

        fig = plt.figure(figsize=(10.8, 5.4), dpi=100, facecolor=BG_DARK)
        gs = gridspec.GridSpec(
            2, 3, width_ratios=[1.15, 1.30, 1.15], height_ratios=[1.0, 1.0],
            wspace=0.28, hspace=0.42, left=0.04, right=0.95, top=0.83, bottom=0.11,
        )
        fig.suptitle(
            "Convolutional Neural Network (28×28 → 6c5 → 2s → 12c5 → 2s → 10)",
            color=TEXT_MAIN, fontsize=13, fontweight="bold", y=0.96,
        )
        acc_str = f"{hist_acc[-1]:.1f}%" if hist_acc else "--"
        loss_str = f"{hist_loss[-1]:.3f}" if hist_loss else "--"
        fig.text(
            0.5, 0.895,
            f"{label_text}   |   Smoothed Loss: {loss_str}   |   Test Accuracy: {acc_str}",
            ha="center", color=CYAN, fontsize=10, fontweight="bold",
        )

        # Col 1 Top: Input '4' vs Live Deconv Reverse Map
        sub_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], wspace=0.12)
        ax_in = fig.add_subplot(sub_gs[0])
        style_axes(ax_in, "Input '4'")
        hide_ticks(ax_in)
        ax_in.imshow(probe_img[:, :, 0], cmap="gray", vmin=0, vmax=1)

        ax_rev = fig.add_subplot(sub_gs[1])
        style_axes(ax_rev, "Conv2 → Image")
        hide_ticks(ax_rev)
        ax_rev.imshow(rev_img, cmap="RdBu_r", vmin=-rev_max, vmax=rev_max)
        ax_rev.set_xlabel("Live Deconv Map", color=TEXT_MUTED, fontsize=8)

        # Col 1 Bottom: Learned Conv1 5x5 kernels
        ax_k = fig.add_subplot(gs[1, 0])
        style_axes(ax_k, "Learned Conv1 Kernels (6 × 5×5)")
        hide_ticks(ax_k)
        ax_k.imshow(tile_images(kernels, 2, 3, pad=1, bg=0.0), cmap="RdBu_r",
                    vmin=-k_max, vmax=k_max, interpolation="nearest")

        # Col 2 Top: Conv1 feature maps (6 x 24x24)
        ax_c1 = fig.add_subplot(gs[0, 1])
        style_axes(ax_c1, "Conv1 Feature Maps (6 × 24×24)")
        hide_ticks(ax_c1)
        ax_c1.imshow(tile_images(c1_maps, 2, 3, pad=2, bg=0.1), cmap="viridis", vmin=0, vmax=1)

        # Col 2 Bottom: Conv2 feature maps (12 x 8x8)
        ax_c2 = fig.add_subplot(gs[1, 1])
        style_axes(ax_c2, "Conv2 Feature Maps (12 × 8×8)")
        hide_ticks(ax_c2)
        ax_c2.imshow(tile_images(c2_maps, 3, 4, pad=1, bg=0.1), cmap="magma", vmin=0, vmax=1)

        # Col 3 Top: Output neuron scores
        ax_out = fig.add_subplot(gs[0, 2])
        style_axes(ax_out, f"Output Neurons (Pred: {pred})")
        colors = [GREEN if k == 4 and pred == 4 else (CORAL if k == pred else CYAN) for k in range(10)]
        ax_out.bar(np.arange(10), probs, color=colors, width=0.72)
        ax_out.set_xticks(range(10))
        ax_out.set_ylim(0, 1.08)
        ax_out.grid(axis="y", color=BORDER_CLR, alpha=0.4, linewidth=0.6)

        # Col 3 Bottom: Loss & Accuracy
        ax_l = fig.add_subplot(gs[1, 2])
        style_axes(ax_l, "Loss & Test Accuracy (%)")
        ax_l.set_xlim(0, numepochs * numbatches)
        ax_l.set_ylim(0, 1.4)
        ax_l.set_xlabel("Mini-batch Step", color=TEXT_MUTED, fontsize=8.5)
        ax_l.grid(color=BORDER_CLR, alpha=0.45, linewidth=0.6)
        if hist_steps:
            ax_l.plot(hist_steps, hist_loss, color=CORAL, linewidth=2.2)
            ax_r = ax_l.twinx()
            ax_r.set_ylim(0, 102)
            ax_r.tick_params(axis="y", colors=GREEN, labelsize=8.5)
            ax_r.spines["right"].set_color(BORDER_CLR)
            ax_r.plot(hist_steps, hist_acc, color=GREEN, linewidth=2.2)

        frames.append(fig_to_pil(fig))
        plt.close(fig)

    err0, _ = cnntest.cnntest(cnn, eval_x, eval_y)
    cnn = cnnff.cnnff(cnn, probe_img)
    hist_steps.append(0)
    hist_loss.append(float(cnn.L) if cnn.L else 1.25)
    hist_acc.append((1.0 - err0) * 100.0)
    render_frame(cnn, "Epoch 0 (Random Initialization)")

    def on_step(net, ep, batch, _loss_val):
        if (ep, batch) in snap_steps:
            err, _ = cnntest.cnntest(net, eval_x, eval_y)
            step = ep * numbatches + batch + 1
            hist_steps.append(step)
            hist_loss.append(float(net.rL[-1]))
            hist_acc.append((1.0 - err) * 100.0)
            render_frame(net, f"Epoch {ep + 1}/{numepochs}  •  Batch {batch + 1}/{numbatches}")

    cnntrain.cnntrain(cnn, cx_train, cy_train, {"alpha": 1.0, "numepochs": numepochs, "batchsize": batchsize},
                      callback=on_step)
    save_gif(frames, os.path.join(ASSETS_DIR, "cnn_training.gif"))


# ---------------------------------------------------------------------------
# 3. DBN Training Animation (with RBM-1 & RBM-2 Reverse Map + Gibbs v2)
# ---------------------------------------------------------------------------
def generate_dbn_gif(train_x, train_y, test_x, test_y):
    print("\n--- Generating DBN training GIF ---")
    probe_idxs = select_probe_digits(test_x, test_y, digits=(0, 3, 5, 8))
    probe_imgs = test_x[probe_idxs]

    np.random.seed(10)
    rbm_opts = {"numepochs": 2, "batchsize": 100, "momentum": 0.0, "alpha": 1.0}
    dbn = DeepBeliefNetwork.DBN(train_x, np.array([100, 100]), rbm_opts)
    numbatches = train_x.shape[0] // rbm_opts["batchsize"]

    rbm1_snaps = {(0, 0), (0, 5), (0, 15), (0, 40), (0, 100), (0, 220), (0, 400), (0, 599),
                  (1, 200), (1, 599)}
    rbm2_snaps = {(0, 50), (0, 299), (0, 599), (1, 599)}
    nn_snaps = {(0, 20), (0, 100), (0, 300), (0, 599), (1, 299), (1, 599)}

    frames = []
    hist_x, hist_recon, hist_acc = [], [], []
    step_counter = [0]

    def rbm_recon(rbm0):
        h_prob = sigm.sigm(np.tile(rbm0.C.T, (probe_imgs.shape[0], 1)) + np.dot(probe_imgs, rbm0.W.T))
        v_prob = sigm.sigm(np.tile(rbm0.B.T, (probe_imgs.shape[0], 1)) + np.dot(h_prob, rbm0.W))
        return v_prob

    def render_frame(phase_title, dbn_obj, acc_val=None):
        rbm0, rbm1 = dbn_obj.RBM[0], dbn_obj.RBM[1]
        rf1 = rbm0.W[:9].reshape(9, 28, 28)
        vmax1 = max(np.percentile(np.abs(rf1), 99), 1e-2)

        # Reverse-map RBM-2 deep units back to 28x28 image space via W2 @ W1
        rf2_proj = (rbm1.W[:9] @ rbm0.W).reshape(9, 28, 28)
        vmax2 = max(np.percentile(np.abs(rf2_proj), 99), 1e-2)

        recon_imgs = rbm_recon(rbm0)

        fig = plt.figure(figsize=(10.8, 5.4), dpi=100, facecolor=BG_DARK)
        gs = gridspec.GridSpec(
            2, 3, width_ratios=[1.15, 1.15, 1.25], height_ratios=[1.0, 1.0],
            wspace=0.28, hspace=0.42, left=0.04, right=0.97, top=0.83, bottom=0.11,
        )
        fig.suptitle(
            "Deep Belief Network (RBM-1 & RBM-2 CD-1 → Unfolded NN)",
            color=TEXT_MAIN, fontsize=13, fontweight="bold", y=0.96,
        )
        err_str = f"{hist_recon[-1]:.1f}" if hist_recon else "--"
        acc_str = f"{acc_val:.1f}%" if acc_val is not None else "Pre-training"
        fig.text(
            0.5, 0.895,
            f"{phase_title}   |   CD Recon Error: {err_str}   |   Test Acc: {acc_str}",
            ha="center", color=CYAN, fontsize=10, fontweight="bold",
        )

        # Col 1 Top: RBM-1 features
        ax_rf1 = fig.add_subplot(gs[0, 0])
        style_axes(ax_rf1, "RBM-1 Stroke Features (3×3)")
        hide_ticks(ax_rf1)
        ax_rf1.imshow(tile_images(rf1, 3, 3, pad=2, bg=0.0), cmap="RdBu_r", vmin=-vmax1, vmax=vmax1)

        # Col 1 Bottom: RBM-2 deep features reverse-mapped to pixel space
        ax_rf2 = fig.add_subplot(gs[1, 0])
        style_axes(ax_rf2, "RBM-2 Reverse Map (W₂·W₁)")
        hide_ticks(ax_rf2)
        ax_rf2.imshow(tile_images(rf2_proj, 3, 3, pad=2, bg=0.0), cmap="RdBu_r", vmin=-vmax2, vmax=vmax2)

        # Col 2 Top: Input digits
        ax_orig = fig.add_subplot(gs[0, 1])
        style_axes(ax_orig, "Visible Input v₁ (0, 3, 5, 8)")
        hide_ticks(ax_orig)
        ax_orig.imshow(tile_images(probe_imgs.reshape(-1, 28, 28), 2, 2, pad=2, bg=0.15),
                       cmap="gray", vmin=0, vmax=1)

        # Col 2 Bottom: Live Gibbs reverse reconstruction v2
        ax_rec = fig.add_subplot(gs[1, 1])
        style_axes(ax_rec, "Live RBM Reconstruction v₂")
        hide_ticks(ax_rec)
        ax_rec.imshow(tile_images(recon_imgs.reshape(-1, 28, 28), 2, 2, pad=2, bg=0.15),
                      cmap="gray", vmin=0, vmax=1)

        # Col 3 Top: RBM reconstruction error
        ax_err = fig.add_subplot(gs[0, 2])
        style_axes(ax_err, "RBM Reconstruction Error")
        if hist_x:
            ax_err.plot(hist_x, hist_recon, color=AMBER, linewidth=2.2)
            ax_err.scatter([hist_x[-1]], [hist_recon[-1]], color=TEXT_MAIN, s=26, zorder=5)
        ax_err.set_xlim(0, len(rbm1_snaps) + len(rbm2_snaps) + len(nn_snaps) + 2)
        ax_err.set_ylim(0, 210)
        ax_err.grid(color=BORDER_CLR, alpha=0.45, linewidth=0.6)

        # Col 3 Bottom: Fine-tuning test accuracy
        ax_acc = fig.add_subplot(gs[1, 2])
        style_axes(ax_acc, "Unfolded NN Test Accuracy (%)")
        acc_pts = [(x, a) for x, a in zip(hist_x, hist_acc) if a is not None]
        if acc_pts:
            xs, ys = zip(*acc_pts)
            ax_acc.plot(xs, ys, color=GREEN, marker="o", markersize=4.5, linewidth=2.2)
        ax_acc.set_xlim(0, len(rbm1_snaps) + len(rbm2_snaps) + len(nn_snaps) + 2)
        ax_acc.set_ylim(0, 102)
        ax_acc.set_xlabel("Training Checkpoint", color=TEXT_MUTED, fontsize=8.5)
        ax_acc.grid(color=BORDER_CLR, alpha=0.45, linewidth=0.6)

        frames.append(fig_to_pil(fig))
        plt.close(fig)

    hist_x.append(0)
    hist_recon.append(196.0)
    hist_acc.append(None)
    render_frame("Phase 1: RBM-1 Untrained (W = 0)", dbn)

    def on_rbm_step(_rbm, rbm_idx, ep, batch, batch_err, _v1, _v2):
        snaps = rbm1_snaps if rbm_idx == 0 else rbm2_snaps
        if (ep, batch) in snaps:
            step_counter[0] += 1
            hist_x.append(step_counter[0])
            hist_recon.append(float(batch_err))
            hist_acc.append(None)
            render_frame(
                f"Phase {rbm_idx + 1}: RBM-{rbm_idx + 1} Epoch {ep + 1}/2 • Batch {batch + 1}/{numbatches}",
                dbn,
            )

    dbn = dbntrain.dbntrain(dbn, train_x, rbm_opts, callback=on_rbm_step)

    nn = dbnunfoldtonn.dbnunfoldtonn(dbn, 10)
    nn.ActivationFunction = "sigm"
    eval_x, eval_y = test_x[:2000], test_y[:2000]

    er0, _ = nntest.nntest(nn, eval_x, eval_y)
    step_counter[0] += 1
    hist_x.append(step_counter[0])
    hist_recon.append(hist_recon[-1])
    hist_acc.append((1.0 - er0) * 100.0)
    render_frame("Phase 3: Unfolded DBN → NN (Before Fine-tuning)", dbn, acc_val=hist_acc[-1])

    def on_nn_step(net, ep, batch, _loss_val, _n):
        if (ep, batch) in nn_snaps:
            er, _ = nntest.nntest(net, eval_x, eval_y)
            acc = (1.0 - er) * 100.0
            step_counter[0] += 1
            hist_x.append(step_counter[0])
            hist_recon.append(hist_recon[-1])
            hist_acc.append(acc)
            dbn.RBM[0].W = net.W[0][:, 1:]
            dbn.RBM[1].W = net.W[1][:, 1:]
            render_frame(
                f"Phase 3: Fine-tuning Epoch {ep + 1}/2 • Batch {batch + 1}/{numbatches}",
                dbn,
                acc_val=acc,
            )

    nntrain.nntrain(nn, train_x, train_y, {"numepochs": 2, "batchsize": 100}, None, None, None,
                    callback=on_nn_step)
    save_gif(frames, os.path.join(ASSETS_DIR, "dbn_training.gif"))


# ---------------------------------------------------------------------------
# 4. SAE Training Animation (Denoising + Encoder/Decoder Reverse Maps)
# ---------------------------------------------------------------------------
def generate_sae_gif(train_x, train_y, test_x, test_y):
    print("\n--- Generating SAE training GIF ---")
    probe_idxs = select_probe_digits(test_x, test_y, digits=(0, 3, 5, 8))
    probe_clean = test_x[probe_idxs]
    rng = np.random.RandomState(42)
    mask = (rng.uniform(0, 1, probe_clean.shape) > 0.5).astype(np.float64)
    probe_noisy = probe_clean * mask

    np.random.seed(0)
    sae = SAE(np.array([784, 100]))
    sae.AutoEncoder[0].ActivationFunction = "sigm"
    sae.AutoEncoder[0].LearningRate = 1.0
    sae.AutoEncoder[0].InputZeroMaskedFraction = 0.5

    numbatches = train_x.shape[0] // 100
    ae_snaps = {(0, 0), (0, 3), (0, 8), (0, 20), (0, 45), (0, 90), (0, 160), (0, 260),
                (0, 400), (0, 599), (1, 150), (1, 350), (1, 599)}
    nn_snaps = {(0, 30), (0, 150), (0, 350), (0, 599), (1, 299), (1, 599)}

    frames = []
    hist_x, hist_ae_loss, hist_acc = [], [], []
    step_counter = [0]

    def render_frame(phase_title, ae_net, acc_val=None):
        ae_net.Testing = True
        ae_net = nnff.nnff(ae_net, probe_noisy, probe_clean)
        ae_net.Testing = False
        recon = ae_net.A[-1].reshape(-1, 28, 28)

        rf_enc = ae_net.W[0][:9, 1:].reshape(9, 28, 28)
        vmax_e = max(np.percentile(np.abs(rf_enc), 99), 1e-2)
        rf_dec = ae_net.W[1][:, 1:10].T.reshape(9, 28, 28)
        vmax_d = max(np.percentile(np.abs(rf_dec), 99), 1e-2)

        panels = []
        for k in range(4):
            panels.extend([probe_noisy[k].reshape(28, 28), probe_clean[k].reshape(28, 28), recon[k]])
        gallery = tile_images(np.array(panels), 4, 3, pad=2, bg=0.15)

        fig = plt.figure(figsize=(10.8, 5.4), dpi=100, facecolor=BG_DARK)
        gs = gridspec.GridSpec(
            2, 3, width_ratios=[1.15, 1.15, 1.25], height_ratios=[1.0, 1.0],
            wspace=0.28, hspace=0.42, left=0.04, right=0.97, top=0.83, bottom=0.11,
        )
        fig.suptitle(
            "Stacked Denoising Autoencoder (50% Masked Input → 100 Hidden → Classifier)",
            color=TEXT_MAIN, fontsize=12.8, fontweight="bold", y=0.96,
        )
        loss_str = f"{hist_ae_loss[-1]:.2f}" if hist_ae_loss else "--"
        acc_str = f"{acc_val:.1f}%" if acc_val is not None else "AE Pre-training"
        fig.text(
            0.5, 0.895,
            f"{phase_title}   |   Denoising Loss: {loss_str}   |   Classifier Acc: {acc_str}",
            ha="center", color=CYAN, fontsize=10, fontweight="bold",
        )

        # Col 1: Noisy | Clean | Live Denoised Reverse Reconstruction
        ax_gal = fig.add_subplot(gs[:, 0])
        style_axes(ax_gal, "Noisy (50%)  |  Clean  |  Live Recon")
        hide_ticks(ax_gal)
        ax_gal.imshow(gallery, cmap="gray", vmin=0, vmax=1)

        # Col 2 Top: Encoder receptive fields W[0]
        ax_enc = fig.add_subplot(gs[0, 1])
        style_axes(ax_enc, "Encoder Weights W[0] (3×3)")
        hide_ticks(ax_enc)
        ax_enc.imshow(tile_images(rf_enc, 3, 3, pad=2, bg=0.0), cmap="RdBu_r", vmin=-vmax_e, vmax=vmax_e)

        # Col 2 Bottom: Decoder reverse templates W[1]^T
        ax_dec = fig.add_subplot(gs[1, 1])
        style_axes(ax_dec, "Decoder Reverse Templates W[1]ᵀ")
        hide_ticks(ax_dec)
        ax_dec.imshow(tile_images(rf_dec, 3, 3, pad=2, bg=0.0), cmap="RdBu_r", vmin=-vmax_d, vmax=vmax_d)

        # Col 3 Top: AE loss
        ax_loss = fig.add_subplot(gs[0, 2])
        style_axes(ax_loss, "Denoising Reconstruction Loss")
        if hist_x:
            ax_loss.plot(hist_x, hist_ae_loss, color=PURPLE, linewidth=2.2)
            ax_loss.scatter([hist_x[-1]], [hist_ae_loss[-1]], color=TEXT_MAIN, s=26, zorder=5)
        ax_loss.set_xlim(0, len(ae_snaps) + len(nn_snaps) + 2)
        ax_loss.set_ylim(0, 105)
        ax_loss.grid(color=BORDER_CLR, alpha=0.45, linewidth=0.6)

        # Col 3 Bottom: Classifier accuracy
        ax_acc = fig.add_subplot(gs[1, 2])
        style_axes(ax_acc, "Initialized NN Test Accuracy (%)")
        acc_pts = [(x, a) for x, a in zip(hist_x, hist_acc) if a is not None]
        if acc_pts:
            xs, ys = zip(*acc_pts)
            ax_acc.plot(xs, ys, color=GREEN, marker="o", markersize=4.5, linewidth=2.2)
        ax_acc.set_xlim(0, len(ae_snaps) + len(nn_snaps) + 2)
        ax_acc.set_ylim(0, 102)
        ax_acc.set_xlabel("Training Checkpoint", color=TEXT_MUTED, fontsize=8.5)
        ax_acc.grid(color=BORDER_CLR, alpha=0.45, linewidth=0.6)

        frames.append(fig_to_pil(fig))
        plt.close(fig)

    ae0 = nnff.nnff(sae.AutoEncoder[0], probe_noisy, probe_clean)
    hist_x.append(0)
    hist_ae_loss.append(float(ae0.L))
    hist_acc.append(None)
    render_frame("Phase 1: Untrained Autoencoder", sae.AutoEncoder[0])

    def on_ae_step(sae_obj, _ae_idx, ep, batch, loss_val, _step):
        if (ep, batch) in ae_snaps:
            step_counter[0] += 1
            hist_x.append(step_counter[0])
            hist_ae_loss.append(float(loss_val))
            hist_acc.append(None)
            render_frame(
                f"Phase 1: Denoising AE Epoch {ep + 1}/2 • Batch {batch + 1}/{numbatches}",
                sae_obj.AutoEncoder[0],
            )

    sae = saetrain(sae, train_x, {"numepochs": 2, "batchsize": 100}, callback=on_ae_step)

    nn = NeuralNetwork.NN(np.array([784, 100, 10]))
    nn.ActivationFunction = "sigm"
    nn.LearningRate = 1.0
    nn.W[0] = sae.AutoEncoder[0].W[0].copy()

    eval_x, eval_y = test_x[:2000], test_y[:2000]
    er0, _ = nntest.nntest(nn, eval_x, eval_y)
    step_counter[0] += 1
    hist_x.append(step_counter[0])
    hist_ae_loss.append(hist_ae_loss[-1])
    hist_acc.append((1.0 - er0) * 100.0)
    render_frame("Phase 2: Initialize NN from AE Weights", sae.AutoEncoder[0], acc_val=hist_acc[-1])

    def on_nn_step(net, ep, batch, _loss_val, _step):
        if (ep, batch) in nn_snaps:
            er, _ = nntest.nntest(net, eval_x, eval_y)
            acc = (1.0 - er) * 100.0
            sae.AutoEncoder[0].W[0] = net.W[0].copy()
            step_counter[0] += 1
            hist_x.append(step_counter[0])
            hist_ae_loss.append(hist_ae_loss[-1])
            hist_acc.append(acc)
            render_frame(
                f"Phase 2: Fine-tuning Epoch {ep + 1}/2 • Batch {batch + 1}/{numbatches}",
                sae.AutoEncoder[0],
                acc_val=acc,
            )

    nntrain.nntrain(nn, train_x, train_y, {"numepochs": 2, "batchsize": 100}, None, None, None,
                    callback=on_nn_step)
    save_gif(frames, os.path.join(ASSETS_DIR, "sae_training.gif"))


# ---------------------------------------------------------------------------
# 5. CAE Training Animation (Conv Encode -> 2x2 Pool -> Unpool -> Live Deconv)
# ---------------------------------------------------------------------------
def generate_cae_gif(train_x, train_y, test_x, test_y):
    print("\n--- Generating CAE training GIF ---")
    cx_train = np.transpose(train_x.reshape(-1, 28, 28), (1, 2, 0))[:, :, :15000]
    cx_test = np.transpose(test_x.reshape(-1, 28, 28), (1, 2, 0))

    probe_idxs = select_probe_digits(test_x, test_y, digits=(0, 3, 5, 8))
    probe_clean = cx_test[:, :, probe_idxs]
    rng = np.random.RandomState(7)
    probe_mask = (rng.uniform(0, 1, probe_clean.shape) > 0.3).astype(np.float64)
    probe_noisy = probe_clean * probe_mask

    np.random.seed(1)
    cae = CAE(inputMaps=1, outputMaps=12, kernelSize=5, scale=2, noise=0.3, momentum=0.8, loss="bce")
    numepochs = 3
    batchsize = 50
    numbatches = cx_train.shape[2] // batchsize
    snap_steps = build_snapshot_steps(numbatches, numepochs)

    frames = []
    hist_steps, hist_loss, hist_mse = [], [], []
    eval_x = cx_test[:, :, :500]

    def render_frame(net, label_text):
        saved_noise = net.InputZeroMaskedFraction
        net.InputZeroMaskedFraction = 0.0
        net = caeff(net, probe_noisy, target=probe_clean, training=False)
        net.InputZeroMaskedFraction = saved_noise

        latent_maps = np.array([net.P[j][:, :, 3] for j in range(12)])  # Digit '8'
        recons = net.O[0]
        enc_k = np.array([net.IK[(0, j)] for j in range(12)])
        dec_k = np.array([net.OK[(0, j)] for j in range(12)])
        k_max = max(np.max(np.abs(enc_k)), np.max(np.abs(dec_k)), 1e-3)

        panels = []
        for idx in range(4):
            panels.extend([probe_noisy[:, :, idx], probe_clean[:, :, idx], recons[:, :, idx]])
        gallery = tile_images(np.array(panels), 4, 3, pad=2, bg=0.15)

        fig = plt.figure(figsize=(10.8, 5.4), dpi=100, facecolor=BG_DARK)
        gs = gridspec.GridSpec(
            2, 3, width_ratios=[1.15, 1.25, 1.2], height_ratios=[1.0, 1.0],
            wspace=0.28, hspace=0.42, left=0.04, right=0.97, top=0.83, bottom=0.11,
        )
        fig.suptitle(
            "Convolutional Autoencoder (30% Noise → 12c5 → 2×2 Pool → Unpool → Deconv)",
            color=TEXT_MAIN, fontsize=12.5, fontweight="bold", y=0.96,
        )
        mse_str = f"{hist_mse[-1]:.4f}" if hist_mse else "--"
        loss_str = f"{hist_loss[-1]:.3f}" if hist_loss else "--"
        fig.text(
            0.5, 0.895,
            f"{label_text}   |   BCE Loss: {loss_str}   |   Test Pixel MSE: {mse_str}",
            ha="center", color=CYAN, fontsize=10, fontweight="bold",
        )

        # Col 1: Noisy | Clean | Live CAE Reverse Map Reconstruction
        ax_gal = fig.add_subplot(gs[:, 0])
        style_axes(ax_gal, "Noisy (30%)  |  Clean  |  Live Deconv")
        hide_ticks(ax_gal)
        ax_gal.imshow(gallery, cmap="gray", vmin=0, vmax=1)

        # Col 2 Top: 12 bottleneck pooled maps (12 x 12x12) for '8'
        ax_lat = fig.add_subplot(gs[0, 1])
        style_axes(ax_lat, "Bottleneck Latent Maps (12 × 12×12)")
        hide_ticks(ax_lat)
        ax_lat.imshow(tile_images(latent_maps, 3, 4, pad=2, bg=0.1), cmap="viridis", vmin=0, vmax=1)

        # Col 2 Bottom: Encoder IK (top 2x6) & Decoder OK (bottom 2x6)
        ax_k = fig.add_subplot(gs[1, 1])
        style_axes(ax_k, "Encoder IK (Top) & Decoder OK (Bottom)")
        hide_ticks(ax_k)
        k_both = np.concatenate([enc_k, dec_k], axis=0)
        ax_k.imshow(tile_images(k_both, 4, 6, pad=1, bg=0.0), cmap="RdBu_r",
                    vmin=-k_max, vmax=k_max, interpolation="nearest")

        # Col 3 Top: BCE training loss
        ax_loss = fig.add_subplot(gs[0, 2])
        style_axes(ax_loss, "Reconstruction BCE Loss")
        if hist_steps:
            ax_loss.plot(hist_steps, hist_loss, color=CORAL, linewidth=2.2)
            ax_loss.scatter([hist_steps[-1]], [hist_loss[-1]], color=TEXT_MAIN, s=26, zorder=5)
        ax_loss.set_xlim(0, numepochs * numbatches)
        ax_loss.set_ylim(0, max(hist_loss[0] * 1.15, 0.25))
        ax_loss.grid(color=BORDER_CLR, alpha=0.45, linewidth=0.6)

        # Col 3 Bottom: Test pixel MSE
        ax_mse = fig.add_subplot(gs[1, 2])
        style_axes(ax_mse, "Test Set Pixel MSE")
        if hist_steps:
            ax_mse.plot(hist_steps, hist_mse, color=GREEN, linewidth=2.2)
            ax_mse.scatter([hist_steps[-1]], [hist_mse[-1]], color=TEXT_MAIN, s=26, zorder=5)
        ax_mse.set_xlim(0, numepochs * numbatches)
        ax_mse.set_ylim(0, max(hist_mse[0] * 1.1, 0.05))
        ax_mse.set_xlabel("Mini-batch Step", color=TEXT_MUTED, fontsize=8.5)
        ax_mse.grid(color=BORDER_CLR, alpha=0.45, linewidth=0.6)

        frames.append(fig_to_pil(fig))
        plt.close(fig)

    mse0, loss0 = caetest(cae, eval_x)
    hist_steps.append(0)
    hist_loss.append(float(loss0))
    hist_mse.append(float(mse0))
    render_frame(cae, "Epoch 0 (Random Initialization)")

    def on_step(net, ep, batch, _loss_val):
        if (ep, batch) in snap_steps:
            mse, _ = caetest(net, eval_x)
            step = ep * numbatches + batch + 1
            hist_steps.append(step)
            hist_loss.append(float(net.rL[-1]))
            hist_mse.append(float(mse))
            render_frame(net, f"Epoch {ep + 1}/{numepochs}  •  Batch {batch + 1}/{numbatches}")

    caetrain(cae, cx_train, {"alpha": 4.0, "numepochs": numepochs, "batchsize": batchsize}, callback=on_step)
    save_gif(frames, os.path.join(ASSETS_DIR, "cae_training.gif"))


def main():
    train_x, train_y, test_x, test_y = load_unified_mnist()
    generate_nn_gif(train_x, train_y, test_x, test_y)
    generate_cnn_gif(train_x, train_y, test_x, test_y)
    generate_dbn_gif(train_x, train_y, test_x, test_y)
    generate_sae_gif(train_x, train_y, test_x, test_y)
    generate_cae_gif(train_x, train_y, test_x, test_y)
    print("\nAll 5 training GIFs generated successfully!")


if __name__ == "__main__":
    main()
