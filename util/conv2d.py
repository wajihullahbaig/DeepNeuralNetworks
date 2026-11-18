# -*- coding: utf-8 -*-
"""
Created on 18th November 2026
Updated on 18th November 2026
@author: Lenovo Laptop

Fast pure-NumPy 2D valid and full convolutions over (H, W, batch) tensors.
Uses sliding_window_view + BLAS matrix multiplication so all input/output
feature map pairs in a layer run in a single vectorized call.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def conv2d_valid_batch(x_maps, kernels, out_maps, rot180=False):
    """
    Compute valid 2D convolution across all (i, j) map pairs.
    x_maps: dict {i: array(H, W, M)} for i in 0..in_maps-1
    kernels: dict {(i, j): array(Kh, Kw)}
    If rot180=True, applies 180-degree rotated kernels (i.e. cross-correlation).
    Returns dict {j: array(H - Kh + 1, W - Kw + 1, M)} for j in 0..out_maps-1.
    """
    in_maps = len(x_maps)
    x_stack = np.stack([x_maps[i] for i in range(in_maps)], axis=0)  # (I, H, W, M)
    first_k = next(iter(kernels.values()))
    Kh, Kw = first_k.shape
    H, W, M = x_stack.shape[1], x_stack.shape[2], x_stack.shape[3]
    Ho, Wo = H - Kh + 1, W - Kw + 1

    # windows: (I, Ho, Wo, M, Kh, Kw) -> (Ho*Wo*M, I*Kh*Kw)
    windows = sliding_window_view(x_stack, (Kh, Kw), axis=(1, 2))
    col = windows.transpose(1, 2, 3, 0, 4, 5).reshape(Ho * Wo * M, in_maps * Kh * Kw)

    # k_mat: (I*Kh*Kw, J)
    k_mat = np.empty((in_maps, Kh, Kw, out_maps), dtype=np.float64)
    for i in range(in_maps):
        for j in range(out_maps):
            k = kernels[(i, j)]
            k_mat[i, :, :, j] = k if rot180 else k[::-1, ::-1]
    k_mat = k_mat.reshape(in_maps * Kh * Kw, out_maps)

    out = (col @ k_mat).reshape(Ho, Wo, M, out_maps)
    return {j: out[:, :, :, j].copy() for j in range(out_maps)}


def conv2d_full_batch(u_maps, kernels, out_maps, rot180=False):
    """
    Compute full 2D convolution across all (i, j) pairs where u_maps is indexed by src
    and kernels is indexed by (src, dst) or (dst, src) via a pre-arranged mapping.
    Here u_maps: dict {src: array(Hu, Wu, M)}, kernels[(src, dst)] -> (Kh, Kw).
    """
    src_maps = len(u_maps)
    u_stack = np.stack([u_maps[s] for s in range(src_maps)], axis=0)  # (S, Hu, Wu, M)
    first_k = next(iter(kernels.values()))
    Kh, Kw = first_k.shape
    Hu, Wu, M = u_stack.shape[1], u_stack.shape[2], u_stack.shape[3]
    Ho, Wo = Hu + Kh - 1, Wu + Kw - 1

    padded = np.pad(
        u_stack,
        ((0, 0), (Kh - 1, Kh - 1), (Kw - 1, Kw - 1), (0, 0)),
        mode="constant",
    )
    windows = sliding_window_view(padded, (Kh, Kw), axis=(1, 2))
    col = windows.transpose(1, 2, 3, 0, 4, 5).reshape(Ho * Wo * M, src_maps * Kh * Kw)

    k_mat = np.empty((src_maps, Kh, Kw, out_maps), dtype=np.float64)
    for s in range(src_maps):
        for d in range(out_maps):
            k = kernels[(s, d)]
            k_mat[s, :, :, d] = k if rot180 else k[::-1, ::-1]
    k_mat = k_mat.reshape(src_maps * Kh * Kw, out_maps)

    out = (col @ k_mat).reshape(Ho, Wo, M, out_maps)
    return {d: out[:, :, :, d].copy() for d in range(out_maps)}


def conv2d_grad_kernels_valid(a_maps, d_maps, Kh, Kw):
    """
    Compute sum over batch of valid correlation between a_maps[i] (H, W, M)
    and d_maps[j] (Ho, Wo, M), matching fftconvolve(flipall(A[i]), D[j], 'valid')[:, :, 0].
    Returns dict {(i, j): array(Kh, Kw)}.
    """
    in_maps = len(a_maps)
    out_maps = len(d_maps)
    a_stack = np.stack([a_maps[i] for i in range(in_maps)], axis=0)  # (I, H, W, M)
    d_stack = np.stack([d_maps[j] for j in range(out_maps)], axis=0)  # (J, Ho, Wo, M)

    windows = sliding_window_view(a_stack, (Kh, Kw), axis=(1, 2))  # (I, Ho, Wo, M, Kh, Kw)
    Ho, Wo, M = d_stack.shape[1], d_stack.shape[2], d_stack.shape[3]

    # X_col: (I*Kh*Kw, Ho*Wo*M), D_mat: (Ho*Wo*M, J)
    x_col = windows.transpose(0, 4, 5, 1, 2, 3).reshape(in_maps * Kh * Kw, Ho * Wo * M)
    d_mat = d_stack.transpose(1, 2, 3, 0).reshape(Ho * Wo * M, out_maps)

    grad = (x_col @ d_mat).reshape(in_maps, Kh, Kw, out_maps)[:, ::-1, ::-1, :]
    return {(i, j): grad[i, :, :, j].copy() for i in range(in_maps) for j in range(out_maps)}


def conv2d_grad_dec_valid(d_dec_maps, u_maps, Kh, Kw):
    """
    Compute gradient of decoder kernels OK[(i, j)] of shape (Kh, Kw) where forward
    full convolution produced d_dec_maps[i] (Hx, Wx, M) from u_maps[j] (Hu, Wu, M).
    Matches fftconvolve(d_dec_maps[i], flipall(u_maps[j]), mode='valid')[:, :, 0].
    """
    in_maps = len(d_dec_maps)
    out_maps = len(u_maps)
    d_stack = np.stack([d_dec_maps[i] for i in range(in_maps)], axis=0)  # (I, Hx, Wx, M)
    u_stack = np.stack([u_maps[j] for j in range(out_maps)], axis=0)  # (J, Hu, Wu, M)
    Hu, Wu, M = u_stack.shape[1], u_stack.shape[2], u_stack.shape[3]

    windows = sliding_window_view(d_stack, (Hu, Wu), axis=(1, 2))  # (I, Kh, Kw, M, Hu, Wu)
    d_col = windows.transpose(0, 1, 2, 4, 5, 3).reshape(in_maps * Kh * Kw, Hu * Wu * M)
    u_mat = u_stack.transpose(1, 2, 3, 0).reshape(Hu * Wu * M, out_maps)

    grad = (d_col @ u_mat).reshape(in_maps, Kh, Kw, out_maps)
    return {(i, j): grad[i, :, :, j].copy() for i in range(in_maps) for j in range(out_maps)}
