import argparse
import io
import math
import os
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from rosbags.highlevel import AnyReader


IMAGE_ASPECT_RATIO = 4 / 3


def quat_to_yaw(x, y, z, w):
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(t3, t4)


def process_scand_img(msg, image_size):
    image = Image.open(io.BytesIO(msg.data.tobytes())).convert("RGB")
    width, height = image.size
    crop_width = min(width, int(height * IMAGE_ASPECT_RATIO))
    left = max((width - crop_width) // 2, 0)
    image = image.crop((left, 0, left + crop_width, height))
    return image.resize(image_size)


def odom_to_xy_yaw(msg, ang_offset=0.0):
    position = msg.pose.pose.position
    orientation = msg.pose.pose.orientation
    yaw = quat_to_yaw(orientation.x, orientation.y, orientation.z, orientation.w) + ang_offset
    return [position.x, position.y], yaw


def is_backwards(pos1, yaw1, pos2, eps=1e-5):
    dx, dy = pos2 - pos1
    return dx * np.cos(yaw1) + dy * np.sin(yaw1) < eps


def filter_backwards(img_list, traj_data, start_slack=0, end_slack=0):
    traj_pos = traj_data["position"]
    traj_yaws = traj_data["yaw"]
    cut_trajs = []
    start = True

    def process_pair(traj_pair):
        new_img_list, new_traj_data = zip(*traj_pair)
        new_traj_data = np.asarray(new_traj_data)
        return (
            new_img_list,
            {
                "position": new_traj_data[:, :2].astype(np.float32),
                "yaw": new_traj_data[:, 2].astype(np.float32),
            },
        )

    for i in range(max(start_slack, 1), len(traj_pos) - end_slack):
        pos1 = traj_pos[i - 1]
        yaw1 = traj_yaws[i - 1]
        pos2 = traj_pos[i]
        if not is_backwards(pos1, yaw1, pos2):
            if start:
                new_traj_pairs = [(img_list[i - 1], [*traj_pos[i - 1], traj_yaws[i - 1]])]
                start = False
            elif i == len(traj_pos) - end_slack - 1:
                cut_trajs.append(process_pair(new_traj_pairs))
            else:
                new_traj_pairs.append((img_list[i - 1], [*traj_pos[i - 1], traj_yaws[i - 1]]))
        elif not start:
            cut_trajs.append(process_pair(new_traj_pairs))
            start = True

    return cut_trajs


def find_topic(reader, topics):
    counts = {connection.topic: connection.msgcount for connection in reader.connections}
    for topic in topics:
        if counts.get(topic, 0) > 0:
            return topic
    return None


def get_images_and_odom(reader, image_topics, odom_topics, rate, image_size):
    image_topic = find_topic(reader, image_topics)
    odom_topic = find_topic(reader, odom_topics)
    if image_topic is None or odom_topic is None:
        return None, None, image_topic, odom_topic

    selected = [connection for connection in reader.connections if connection.topic in {image_topic, odom_topic}]
    synced_images = []
    positions = []
    yaws = []
    current_image = None
    current_odom = None
    current_time = reader.start_time / 1e9

    for connection, timestamp, rawdata in reader.messages(connections=selected):
        msg = reader.deserialize(rawdata, connection.msgtype)
        if connection.topic == image_topic:
            current_image = msg
        elif connection.topic == odom_topic:
            current_odom = msg

        timestamp_sec = timestamp / 1e9
        if timestamp_sec - current_time >= 1.0 / rate:
            if current_image is not None and current_odom is not None:
                synced_images.append(process_scand_img(current_image, image_size))
                position, yaw = odom_to_xy_yaw(current_odom)
                positions.append(position)
                yaws.append(yaw)
                current_time = timestamp_sec

    traj_data = {
        "position": np.asarray(positions, dtype=np.float32),
        "yaw": np.asarray(yaws, dtype=np.float32),
    }
    return synced_images, traj_data, image_topic, odom_topic


def trajectory_stem(bag_path):
    parent = bag_path.parent.name
    stem = bag_path.stem
    return f"{parent}_{stem}"


def process_bag(bag_path, output_dir, rate, image_size):
    with AnyReader([bag_path]) as reader:
        images, traj_data, image_topic, odom_topic = get_images_and_odom(
            reader,
            image_topics=["/image_raw/compressed", "/camera/rgb/image_raw/compressed"],
            odom_topics=["/odom", "/jackal_velocity_controller/odom"],
            rate=rate,
            image_size=image_size,
        )

    if images is None or traj_data is None:
        print(f"SKIP {bag_path}: missing image/odom topic (image={image_topic}, odom={odom_topic})")
        return []

    cut_trajs = filter_backwards(images, traj_data)
    saved = []
    base_name = trajectory_stem(bag_path)

    for segment_index, (segment_images, segment_traj_data) in enumerate(cut_trajs):
        traj_name = f"{base_name}_{segment_index}"
        traj_dir = output_dir / traj_name
        traj_dir.mkdir(parents=True, exist_ok=True)

        with (traj_dir / "traj_data.pkl").open("wb") as f:
            pickle.dump(segment_traj_data, f)

        for frame_index, image in enumerate(segment_images):
            image.save(traj_dir / f"{frame_index}.jpg")

        saved.append((traj_name, len(segment_images)))

    return saved


def iter_bag_files(input_dir, num_trajs):
    bag_files = sorted(Path(input_dir).rglob("*.bag"))
    if num_trajs >= 0:
        bag_files = bag_files[:num_trajs]
    return bag_files


def main():
    parser = argparse.ArgumentParser(description="Process SCAND ROS1 bags into NWM/ViNT trajectory folders.")
    parser.add_argument("--input-dir", required=True, help="Directory containing SCAND .bag files.")
    parser.add_argument("--output-dir", required=True, help="Processed output directory, e.g. datasets/scand_320.")
    parser.add_argument("--num-trajs", type=int, default=-1, help="Number of bags to process after sorting; -1 for all.")
    parser.add_argument("--sample-rate", type=float, default=4.0, help="Sampling rate in Hz.")
    parser.add_argument("--image-width", type=int, default=320)
    parser.add_argument("--image-height", type=int, default=240)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image_size = (args.image_width, args.image_height)

    total_saved = []
    for bag_path in iter_bag_files(args.input_dir, args.num_trajs):
        saved = process_bag(bag_path, output_dir, args.sample_rate, image_size)
        if saved:
            joined = ", ".join(f"{name}:{length}" for name, length in saved)
            print(f"SAVED {bag_path}: {joined}")
            total_saved.extend(saved)
        else:
            print(f"SAVED {bag_path}: no forward segments")

    print(f"Processed {len(total_saved)} trajectory segments into {os.fspath(output_dir)}")


if __name__ == "__main__":
    main()
