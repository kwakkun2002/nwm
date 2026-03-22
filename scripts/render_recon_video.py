#!/usr/bin/env python3
"""Render RECON HDF5 session(s) to a video file.

Usage:
    python scripts/render_recon_video.py --input path/to/session_folder --output session.mp4
    python scripts/render_recon_video.py --input file1.hdf5 file2.hdf5 --output out.mp4 --fps 10
    python scripts/render_recon_video.py --input "datasets/recon_raw/recon_release/jackal_2019-08-02*"
"""
import argparse
import glob
import io
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import shutil
import subprocess

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

HAS_FFMPEG = shutil.which('ffmpeg') is not None

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


def bytes2im(value):
    return np.array(Image.open(io.BytesIO(bytes(value))))


def collect_hdf5_files(inputs):
    fnames = []
    for inp in inputs:
        if os.path.isdir(inp):
            fnames += sorted(glob.glob(os.path.join(inp, '*.hdf5')))
        elif os.path.isfile(inp):
            fnames.append(inp)
        else:
            # treat as glob pattern
            fnames += sorted(glob.glob(inp))
    seen = set()
    result = []
    for f in fnames:
        if f not in seen:
            seen.add(f)
            result.append(f)
    return result


def render_frame(f, t, full_gps_track, fig, axes):
    """Render timestep t from open HDF5 file f. Returns (H, W, 3) uint8 frame."""
    ((ax_rgb_left, ax_rgb_right, ax_thermal, ax_lidar),
     (ax_gps, ax_coll, ax_imu, ax_vel)) = axes

    for ax in fig.get_axes():
        ax.cla()

    # --- RGB cameras ---
    ax_rgb_left.imshow(bytes2im(f['images/rgb_left'][t]))
    ax_rgb_left.set_title('Left Camera', fontsize=8)
    ax_rgb_left.axis('off')

    ax_rgb_right.imshow(bytes2im(f['images/rgb_right'][t]))
    illum = float(f['android/illuminance'][t])
    ax_rgb_right.set_title(f'Right Camera  ({illum:.0f} lux)', fontsize=8)
    ax_rgb_right.axis('off')

    # --- Thermal ---
    thermal = f['images/thermal'][t]
    ax_thermal.imshow(thermal, cmap='inferno')
    ax_thermal.set_title('Thermal', fontsize=8)
    ax_thermal.axis('off')

    # --- LiDAR: front 120° arc (±60° from forward) ---
    lidar = f['lidar'][t].copy()
    meas_idx = list(range(300, 360)) + list(range(0, 60))
    angles = np.deg2rad(np.linspace(30, 150., 120))
    lr = lidar[meas_idx].copy()
    notfinite = ~np.isfinite(lr)
    lr[notfinite] = 15.0
    colors = ['r' if nf else 'steelblue' for nf in notfinite]
    ax_lidar.scatter(lr * np.cos(angles), lr * np.sin(angles), c=colors, s=3)
    ax_lidar.set_xlim(-16, 16)
    ax_lidar.set_ylim(0, 16)
    ax_lidar.set_aspect('equal')
    ax_lidar.set_title('LiDAR front 120°', fontsize=8)
    ax_lidar.tick_params(labelsize=6)

    # --- GPS track: full session as background, current position as red dot ---
    latlong = f['gps/latlong'][t]
    is_fixed = bool(f['gps/is_fixed'][t]) if 'gps/is_fixed' in f else True
    if len(full_gps_track) > 1:
        lls = np.array(full_gps_track)
        ax_gps.plot(lls[:, 1], lls[:, 0], 'b.', ms=1, alpha=0.3)
    if is_fixed and np.isfinite(latlong).all():
        ax_gps.plot(latlong[1], latlong[0], 'ro', ms=6)
    ax_gps.set_title('GPS Track', fontsize=8)
    ax_gps.tick_params(labelsize=5)

    # --- Collision flags ---
    coll_keys = ['any', 'physical', 'close', 'flipped', 'stuck', 'outside_geofence']
    short_labels = ['any', 'phys', 'close', 'flip', 'stuck', 'geo']
    colls = [float(f[f'collision/{k}'][t]) for k in coll_keys]
    ax_coll.bar(np.arange(len(colls)), colls, color='crimson')
    ax_coll.set_xticks(np.arange(len(colls)))
    ax_coll.set_xticklabels(short_labels, fontsize=6)
    ax_coll.set_ylim(0, 1)
    ax_coll.set_title('Collision', fontsize=8)

    # --- IMU ---
    linacc = f['imu/linear_acceleration'][t].copy()
    angvel_imu = f['imu/angular_velocity'][t].copy()
    linacc[2] = -linacc[2] + 9.81  # flip z (sensor mounted upside-down) and remove gravity
    imu_vals = np.concatenate([linacc, angvel_imu])
    ax_imu.bar(np.arange(6), np.clip(imu_vals, -3, 3), color='dimgray', width=0.8)
    ax_imu.set_xticks(np.arange(6))
    ax_imu.set_xticklabels(['ax', 'ay', 'az', 'wx', 'wy', 'wz'], fontsize=6)
    ax_imu.set_ylim(-3, 3)
    ax_imu.set_title('IMU', fontsize=8)

    # --- Speed / steer ---
    cmd_v = float(f['commands/linear_velocity'][t])
    cmd_w = -float(f['commands/angular_velocity'][t])
    act_v = float(f['jackal/linear_velocity'][t])
    act_w = -float(f['jackal/angular_velocity'][t])
    ax_vel.plot([0, 0], [0, cmd_v], '-k', lw=10, label='Cmd')
    ax_vel.plot([0, cmd_w], [0, 0], '-k', lw=10)
    ax_vel.plot([0, 0], [0, act_v], '-c', lw=4, label='Act')
    ax_vel.plot([0, act_w], [0, 0], '-c', lw=4)
    ax_vel.set_xlim(-1.5, 1.5)
    ax_vel.set_ylim(-1.5, 1.5)
    ax_vel.legend(fontsize=6, loc='lower left')
    ax_vel.set_title('Speed / Steer', fontsize=8)

    T = len(f['collision/any'])
    fname = os.path.basename(f.filename)
    fig.suptitle(f'{fname}   t={t + 1}/{T}', fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.96], pad=0.4)

    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def preload_gps_track(hdf5_files):
    """Collect all valid GPS points across the session for the background track."""
    track = []
    for fname in hdf5_files:
        with h5py.File(fname, 'r') as f:
            lls = f['gps/latlong'][:]
            if 'gps/is_fixed' in f:
                fixed = f['gps/is_fixed'][:]
            else:
                fixed = np.ones(len(lls), dtype=bool)
            for i in range(len(lls)):
                if fixed[i] and np.isfinite(lls[i]).all():
                    track.append(lls[i])
    return track


def count_total_frames(hdf5_files):
    total = 0
    for fname in hdf5_files:
        with h5py.File(fname, 'r') as f:
            total += len(f['collision/any'])
    return total


def make_writer(output_path, fps, frame_shape):
    """Return (writer_type, writer_obj)."""
    h, w = frame_shape[:2]

    if HAS_IMAGEIO:
        try:
            writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=7,
                                        output_params=['-pix_fmt', 'yuv420p'])
            return 'imageio', writer
        except Exception as e:
            print(f'[imageio] failed: {e}', file=sys.stderr)

    if HAS_FFMPEG:
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',
            '-pix_fmt', 'rgb24',
            '-r', str(fps),
            '-i', 'pipe:0',
            '-vcodec', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            output_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return 'ffmpeg', proc

    if HAS_CV2:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
        if writer.isOpened():
            return 'cv2', writer

    return None, None


def write_frame(writer_type, writer, frame):
    if writer_type == 'imageio':
        writer.append_data(frame)
    elif writer_type == 'ffmpeg':
        writer.stdin.write(frame.tobytes())
    else:  # cv2
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def close_writer(writer_type, writer):
    if writer_type == 'imageio':
        writer.close()
    elif writer_type == 'ffmpeg':
        writer.stdin.close()
        writer.wait()
    else:  # cv2
        writer.release()


def main():
    parser = argparse.ArgumentParser(
        description='Render RECON HDF5 session(s) to an MP4 video',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--input', nargs='+', required=True,
                        help='HDF5 file(s), folder(s), or glob pattern(s)')
    parser.add_argument('--output', default='recon_video.mp4',
                        help='Output video path (default: recon_video.mp4)')
    parser.add_argument('--fps', type=float, default=10.0,
                        help='Video frame rate (default: 10 to match ~10 Hz data)')
    parser.add_argument('--width', type=int, default=1600,
                        help='Output frame width in pixels (default: 1600)')
    parser.add_argument('--height', type=int, default=800,
                        help='Output frame height in pixels (default: 800)')
    args = parser.parse_args()

    if not HAS_IMAGEIO and not HAS_FFMPEG and not HAS_CV2:
        print('ERROR: need imageio[ffmpeg], ffmpeg in PATH, or opencv-python.', file=sys.stderr)
        sys.exit(1)

    hdf5_files = collect_hdf5_files(args.input)
    if not hdf5_files:
        print(f'ERROR: no .hdf5 files found for: {args.input}', file=sys.stderr)
        sys.exit(1)

    print(f'Sessions  : {len(hdf5_files)} file(s)')
    print(f'Output    : {args.output}')
    print(f'FPS       : {args.fps}')

    print('Pre-loading GPS track...')
    full_gps_track = preload_gps_track(hdf5_files)
    print(f'  {len(full_gps_track)} valid GPS fixes')

    total_frames = count_total_frames(hdf5_files)
    print(f'Total frames: {total_frames}')

    dpi = 100
    fig, axes = plt.subplots(2, 4, figsize=(args.width / dpi, args.height / dpi), dpi=dpi)

    writer_type, writer = None, None
    frame_count = 0

    iter_files = hdf5_files
    if HAS_TQDM:
        iter_files = tqdm(hdf5_files, desc='Files', unit='file')

    try:
        for fname in iter_files:
            with h5py.File(fname, 'r') as f:
                T = len(f['collision/any'])
                t_range = range(T)
                if HAS_TQDM and len(hdf5_files) == 1:
                    t_range = tqdm(t_range, desc='Frames', unit='frame')
                for t in t_range:
                    frame = render_frame(f, t, full_gps_track, fig, axes)

                    if writer is None:
                        writer_type, writer = make_writer(args.output, args.fps, frame.shape)
                        if writer is None:
                            print('ERROR: could not create video writer.', file=sys.stderr)
                            sys.exit(1)
                        print(f'Writer    : {writer_type}')

                    write_frame(writer_type, writer, frame)

                    frame_count += 1
                    if not HAS_TQDM and frame_count % 100 == 0:
                        print(f'  {frame_count}/{total_frames} frames', end='\r', flush=True)
    finally:
        plt.close(fig)
        if writer is not None:
            close_writer(writer_type, writer)

    print(f'\nDone: {frame_count} frames → {args.output}')


if __name__ == '__main__':
    main()
