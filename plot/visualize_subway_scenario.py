import argparse
import glob
import math
import os
import sys

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.collections import LineCollection

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from subway_station_scenarios.generate_scenarios import (
        SubwayStationScenarioGenerator as SubwayStationScenarioGeneratorV1,
    )
except ImportError:
    SubwayStationScenarioGeneratorV1 = None

try:
    from subway_station_scenarios.generate_scenarios import (
        SubwayStationScenarioGenerator as SubwayStationScenarioGeneratorV2,
    )
except ImportError:
    SubwayStationScenarioGeneratorV2 = None


def _select_generator_class(osm_file):
    norm_path = os.path.normpath(osm_file).lower()
    if "subway_station_scenarios" in norm_path or os.path.basename(norm_path) == "subway_station.osm":
        if SubwayStationScenarioGeneratorV2 is not None:
            return SubwayStationScenarioGeneratorV2
    if SubwayStationScenarioGeneratorV1 is not None:
        return SubwayStationScenarioGeneratorV1
    if SubwayStationScenarioGeneratorV2 is not None:
        return SubwayStationScenarioGeneratorV2
    raise ImportError("Cannot import a subway-station scenario generator.")


def _portal_coords(generator_cls):
    coords = {
        "subway": generator_cls.SUBWAY_CENTER["subway"],
    }
    for name, coord_list in generator_cls.ROAD_ANCHORS.items():
        if coord_list:
            coords[name] = coord_list[0]
    return coords


def _lon_to_meters(lon: float, reference_lon: float, center_lat: float) -> float:
    earth_radius = 6371000.0
    lon_diff_rad = math.radians(lon - reference_lon)
    return lon_diff_rad * earth_radius * math.cos(math.radians(center_lat))


def _lat_to_meters(lat: float, reference_lat: float) -> float:
    earth_radius = 6371000.0
    lat_diff_rad = math.radians(lat - reference_lat)
    return lat_diff_rad * earth_radius


def load_and_project_network(osm_file):
    print(f"Loading subway walkable network from {osm_file}...")
    generator_cls = _select_generator_class(osm_file)
    generator = generator_cls(osm_file)
    generator.load_osm_data()

    lines = []
    seen_edges = set()
    for u, v in generator.osm_graph.edges():
        edge_key = tuple(sorted((u, v)))
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)
        x1, y1 = generator.node_positions_cache[u]
        x2, y2 = generator.node_positions_cache[v]
        lines.append([(x1, y1), (x2, y2)])

    entrance_pts = []
    for group_name, (lat, lon) in _portal_coords(generator_cls).items():
        radius = 70.0 if group_name == "subway" else 90.0
        entrance_nodes = generator._find_nodes_near_coordinate(lat, lon, radius_m=radius)
        if entrance_nodes:
            coords = [generator.node_positions_cache[node_id] for node_id in entrance_nodes]
            avg_x = sum(point[0] for point in coords) / len(coords)
            avg_y = sum(point[1] for point in coords) / len(coords)
            entrance_pts.append((avg_x, avg_y))

    return lines, entrance_pts


def check_scenario_files():
    search_path = "generated_scenarios/subway_station_max_ue_*/*.yaml"
    files = glob.glob(search_path)
    if not files:
        print("No subway station scenario files found in generated_scenarios/subway_station_max_ue_*")
        return None
    return sorted(files)[0]


def visualize_environment(
    scenario_file=None,
    osm_file="subway_station/subway_station.osm",
    save_path=None,
    frame_stride=1,
    show_empty_tail=False,
    fps=5,
    dpi=90,
    tail_length=15,
):
    if not scenario_file:
        scenario_file = check_scenario_files()
        if not scenario_file:
            return

    print(f"Visualizing Scenario: {scenario_file}")
    with open(scenario_file, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(right=0.75)

    entrance_pts = []
    if os.path.exists(osm_file):
        lines, entrance_pts = load_and_project_network(osm_file)
        lc = LineCollection(lines, colors="gray", linewidths=0.7, alpha=0.8, zorder=1)
        ax.add_collection(lc)
    else:
        print(f"Warning: OSM file {osm_file} not found. Skipped drawing roads.")

    width = config["scene"]["width"]
    height = config["scene"]["height"]
    max_steps = config["scene"]["total_steps"]

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.set_title("Subway Station Scenario", fontsize=16)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")

    import matplotlib.patches as patches

    for ex, ey in entrance_pts:
        rect_size = 40
        rect = patches.Rectangle(
            (ex - rect_size / 2, ey - rect_size / 2),
            rect_size,
            rect_size,
            linewidth=2.0,
            edgecolor="gold",
            facecolor="gold",
            alpha=0.3,
            zorder=10,
        )
        ax.add_patch(rect)

    for bs in config.get("base_stations", []):
        ax.plot(
            bs["position"][0],
            bs["position"][1],
            "s",
            color="red",
            markersize=12,
            markeredgecolor="darkred",
            markeredgewidth=2,
            zorder=5,
        )
        ax.text(
            bs["position"][0],
            bs["position"][1] + 20,
            f"BS{bs['id']}",
            ha="center",
            va="bottom",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"),
            zorder=6,
        )

    info_text = ax.text(
        1.05,
        0.70,
        "",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.5"),
        verticalalignment="top",
        linespacing=1.4,
    )

    legend_elements = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="red", markersize=10, label="Base Station"),
        patches.Rectangle((0, 0), 1, 1, facecolor="gold", edgecolor="gold", alpha=0.3, label="Entrance Area"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=8, label="Active User"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.00, 1.00),
        fancybox=True,
        shadow=True,
        fontsize=10,
        ncol=1,
    )

    step_to_users = {step: [] for step in range(max_steps + 1)}
    user_full_traj = {}

    for dev in config.get("mobile_devices", []):
        dev_id = dev["id"]
        user_full_traj[dev_id] = {}
        for point in dev.get("trajectory", []):
            step = int(point.get("step", point.get("time", 0)))
            pos = point["position"]
            user_full_traj[dev_id][step] = pos
            if step <= max_steps:
                step_to_users[step].append({"id": dev_id, "position": pos})

    last_nonempty_frame = max(
        (step for step, users in step_to_users.items() if users),
        default=0,
    )
    effective_max_frame = max_steps if show_empty_tail else last_nonempty_frame
    num_devices = len(config.get("mobile_devices", []))

    if save_path and save_path.lower().endswith(".gif") and frame_stride == 1 and effective_max_frame > 300:
        frame_stride = max(2, math.ceil((effective_max_frame + 1) / 180))
        print(
            f"Using frame_stride={frame_stride} for GIF export to reduce export time and file size."
        )
    if save_path and save_path.lower().endswith(".gif") and num_devices > 100 and frame_stride < 3:
        frame_stride = 3
        print("Large-user GIF detected; bumping frame_stride to 3 for faster export.")
    if save_path and tail_length > 10 and num_devices > 120:
        tail_length = 10
        print("Large-user export detected; reducing tail_length to 10.")

    colors = ["blue", "orange", "purple", "brown", "pink", "olive", "cyan", "magenta"]
    tail_collection = LineCollection([], linewidths=2, alpha=0.7, zorder=2)
    ax.add_collection(tail_collection)
    user_scatter = ax.scatter(
        [],
        [],
        c="green",
        s=140,
        edgecolors="black",
        linewidths=1,
        zorder=3,
    )

    def init():
        tail_collection.set_segments([])
        tail_collection.set_color([])
        user_scatter.set_offsets(np.empty((0, 2)))
        info_text.set_text("")
        return tail_collection, user_scatter, info_text

    def update(frame):
        users_at_frame = step_to_users.get(frame, [])
        tail_segments = []
        tail_colors = []
        scatter_offsets = []

        for user_data in users_at_frame:
            user_id = user_data["id"]
            curr_pos = user_data["position"]
            color = colors[user_id % len(colors)]
            scatter_offsets.append(curr_pos)

            history_pts = []
            for hist_step in range(max(0, frame - tail_length), frame + 1):
                if hist_step in user_full_traj[user_id]:
                    history_pts.append(user_full_traj[user_id][hist_step])

            if len(history_pts) > 1:
                tail_segments.append(history_pts)
                tail_colors.append(color)

        tail_collection.set_segments(tail_segments)
        tail_collection.set_color(tail_colors if tail_colors else [])
        user_scatter.set_offsets(np.array(scatter_offsets) if scatter_offsets else np.empty((0, 2)))

        ax.set_title(
            f"Subway Station Scenario - Step: {frame}/{effective_max_frame}",
            fontsize=16,
        )
        info_text.set_text(f"Active Users: {len(users_at_frame)}")

        return tail_collection, user_scatter, info_text

    frame_sequence = list(range(0, effective_max_frame + 1, max(1, frame_stride)))
    print(
        f"Creating animation for {len(frame_sequence)} frames "
        f"(last_nonempty_frame={last_nonempty_frame}, frame_stride={frame_stride})..."
    )
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frame_sequence,
        init_func=init,
        blit=False,
        interval=200,
        repeat=False,
        cache_frame_data=False,
    )

    plt.tight_layout()

    if save_path:
        print(f"Saving animation to {save_path}...")
        if save_path.endswith(".mp4"):
            writer = animation.FFMpegWriter(fps=fps)
        else:
            writer = animation.PillowWriter(fps=fps)
            if max_steps > 300:
                print("GIF export can be slow for long animations; MP4 is usually much faster.")

        def report_progress(current_frame, total_frames):
            if current_frame == 0 or (current_frame + 1) % 25 == 0 or current_frame + 1 == total_frames:
                print(f"Saving frame {current_frame + 1}/{total_frames}...")

        ani.save(save_path, writer=writer, dpi=dpi, progress_callback=report_progress)
        print("Save completed!")
        plt.close(fig)
    else:
        plt.show()
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize subway station ScalableMEC scenarios")
    parser.add_argument("--scenario", type=str, default=None, help="Specific YAML scenario to load")
    parser.add_argument(
        "--osm_file",
        type=str,
        default="subway_station/subway_station.osm",
        help="OSM file used as the background map",
    )
    parser.add_argument("--save", type=str, default=None, help="Path to save the animation")
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=1,
        help="Use every N-th frame to speed up export or playback.",
    )
    parser.add_argument(
        "--show_empty_tail",
        action="store_true",
        help="Keep playing until scene.total_steps even if all users have already left.",
    )
    parser.add_argument("--fps", type=int, default=5, help="Output animation fps when saving.")
    parser.add_argument("--dpi", type=int, default=90, help="Output dpi when saving.")
    parser.add_argument(
        "--tail_length",
        type=int,
        default=15,
        help="Tail length in frames for each user's recent trajectory.",
    )
    args = parser.parse_args()

    visualize_environment(
        scenario_file=args.scenario,
        osm_file=args.osm_file,
        save_path=args.save,
        frame_stride=max(1, args.frame_stride),
        show_empty_tail=args.show_empty_tail,
        fps=max(1, args.fps),
        dpi=max(60, args.dpi),
        tail_length=max(1, args.tail_length),
    )
