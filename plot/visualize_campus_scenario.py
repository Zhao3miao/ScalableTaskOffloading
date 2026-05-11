import yaml
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import numpy as np
import osmnx as ox
import math
import glob
import os
import argparse

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"

def _lon_to_meters(lon: float, reference_lon: float, center_lat: float) -> float:
    earth_radius = 6371000.0
    lon_diff_rad = math.radians(lon - reference_lon)
    return lon_diff_rad * earth_radius * math.cos(math.radians(center_lat))

def _lat_to_meters(lat: float, reference_lat: float) -> float:
    earth_radius = 6371000.0
    lat_diff_rad = math.radians(lat - reference_lat)
    return lat_diff_rad * earth_radius

def load_and_project_network(osm_file):
    print(f"Loading OSM network from {osm_file}...")
    G = ox.graph_from_xml(osm_file)
    G = ox.utils_graph.get_undirected(G)
    nodes, edges = ox.graph_to_gdfs(G)
    
    min_lon, min_lat = nodes.geometry.x.min(), nodes.geometry.y.min()
    max_lon, max_lat = nodes.geometry.x.max(), nodes.geometry.y.max()
    center_lat = (min_lat + max_lat) / 2.0
    
    # Project nodes to local metric coordinates
    node_positions = {}
    for node_id, node_data in nodes.iterrows():
        lon = float(node_data.geometry.x)
        lat = float(node_data.geometry.y)
        x = _lon_to_meters(lon, min_lon, center_lat)
        y = _lat_to_meters(lat, min_lat)
        node_positions[node_id] = (x, y)
        
    # Build line segments for rendering
    lines = []
    for _, edge_data in edges.iterrows():
        if hasattr(edge_data.geometry, "coords"):
            coords = list(edge_data.geometry.coords)
            projected_line = []
            for lon, lat in coords:
                x = _lon_to_meters(float(lon), min_lon, center_lat)
                y = _lat_to_meters(float(lat), min_lat)
                projected_line.append((x, y))
            lines.append(projected_line)
            
    # Also expose specific gate nodes
    GATE_COORDS = {
        "West_Gate": (36.543179, 116.822931),
        "North_Gate": (36.551051, 116.828600)
    }
    gate_pts = []
    for name, coords in GATE_COORDS.items():
        lat, lon = coords
        x = _lon_to_meters(lon, min_lon, center_lat)
        y = _lat_to_meters(lat, min_lat)
        gate_pts.append((x, y))
        
    return lines, gate_pts, node_positions

def check_scenario_files():
    search_path = "generated_scenarios/campus_max_ue_*/*.yaml"
    files = glob.glob(search_path)
    if not files:
        print("No scenario files found in generated_scenarios/campus_max_ue_*")
        return None
    return sorted(files)[0]

def visualize_environment(scenario_file=None, osm_file="campus_scenarios/sdnu.osm", save_path=None):
    if not scenario_file:
        scenario_file = check_scenario_files()
        if not scenario_file: return
        
    print(f"Visualizing Scenario: {scenario_file}")
    with open(scenario_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(right=0.75)
    
    # Draw Background Map Network
    if os.path.exists(osm_file):
        lines, gate_pts, _ = load_and_project_network(osm_file)
        lc = LineCollection(lines, colors='gray', linewidths=0.5, alpha=0.7, zorder=1)
        ax.add_collection(lc)
        
        # Plot physical gates
        # if gate_pts:
        #     gx, gy = zip(*gate_pts)
        #     ax.scatter(gx, gy, c='green', marker='*', s=300, edgecolor='black', zorder=4, label='Campus Gates')
    else:
        print(f"Warning: OSM file {osm_file} not found. Skipped drawing roads.")

    # Base dimensions
    width = config['scene']['width']
    height = config['scene']['height']
    max_steps = config['scene']['total_steps']
    
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect("equal")
    ax.set_title("Real Scenario CIO Load Balancing", fontsize=16)
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    
    import matplotlib.patches as patches

    if 'gate_pts' in locals() and gate_pts:
        gate_names = ["West Gate", "North Gate"]
        for idx, (gx, gy) in enumerate(gate_pts):
            name = gate_names[idx] if idx < len(gate_names) else "Gate"
            # Draw a 60x60 meter rectangle centered at the gate coordinate
            rect_size = 60
            rect = patches.Rectangle(
                (gx - rect_size/2, gy - rect_size/2), 
                rect_size, rect_size, 
                linewidth=2.5, edgecolor='gold', facecolor='gold', alpha=0.3, zorder=10
            )
            ax.add_patch(rect)
            # Text removed per user request: ax.text(gx, gy - 35, ...)

    # Draw Edge Servers
    bs_data = config.get('base_stations', [])
    for bs in bs_data:
        ax.plot(bs['position'][0], bs['position'][1], "s", color="red", markersize=12, markeredgecolor="darkred", markeredgewidth=2, zorder=5)
        ax.text(bs['position'][0], bs['position'][1] + 20, f"BS{bs['id']}", ha="center", va="bottom", fontsize=9, bbox=dict(facecolor="white", alpha=0.8, boxstyle="round,pad=0.3"), zorder=6)
        
    # Setup User scatter (using lines for trajectory tails and scatter for current pos inside the update function)
    # We will remove the static dot scatter and just return a list of artists per frame.
    
    # Info Text box on the right
    info_text = ax.text(1.05, 0.70, '', transform=ax.transAxes, fontsize=11, bbox=dict(facecolor="white", alpha=0.9, boxstyle="round,pad=0.5"), verticalalignment="top", linespacing=1.4)

    legend_elements = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="red", markersize=10, label="Base Station"),
        patches.Rectangle((0, 0), 1, 1, facecolor="gold", edgecolor="gold", alpha=0.3, label="Gate Area"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="green", markersize=8, label="Active User"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.00, 1.00), fancybox=True, shadow=True, fontsize=10, ncol=1)

    # Process user trajectories into an easily querable dictionary: step -> [positions and dev_id]

    step_to_users = {step: [] for step in range(max_steps + 1)}
    
    # Pre-parse trajectories for tails
    user_full_traj = {} # dev_id -> full dict of {step: position}
    
    for dev in config.get('mobile_devices', []):
        dev_id = dev['id']
        user_full_traj[dev_id] = {}
        for pt in dev.get('trajectory', []):
            s = int(pt.get('step', pt.get('time', 0)))
            pos = pt['position']
            user_full_traj[dev_id][s] = pos
            if s <= max_steps:
                step_to_users[s].append({"id": dev_id, "position": pos})

    colors = ["blue", "orange", "purple", "brown", "pink", "olive", "cyan", "magenta"]
    tail_length = 15
    max_tail_jump_m = 120.0

    dynamic_artists = []

    def init():
        return ()

    def recent_contiguous_tail(uid, frame):
        if frame not in user_full_traj[uid]:
            return []

        tail = [user_full_traj[uid][frame]]
        previous_pos = tail[0]
        for h_step in range(frame - 1, max(-1, frame - tail_length - 1), -1):
            if h_step not in user_full_traj[uid]:
                break

            pos = user_full_traj[uid][h_step]
            jump = math.hypot(previous_pos[0] - pos[0], previous_pos[1] - pos[1])
            if jump > max_tail_jump_m:
                break

            tail.append(pos)
            previous_pos = pos

        tail.reverse()
        return tail

    def update(frame):
        # Clear previous frame's dynamic lines and points
        for art in dynamic_artists:
            try:
                art.remove()
            except Exception:
                pass
        dynamic_artists.clear()

        users_at_frame = step_to_users.get(frame, [])
        
        # 1. Draw Trajectories directly onto ax, append to dynamic_artists
        for ud in users_at_frame:
            uid = ud['id']
            curr_pos = ud['position']
            color = colors[uid % len(colors)]
            
            # Draw only the current contiguous trajectory segment.
            # If a UE leaves and later reappears elsewhere with the same id,
            # the tail is reset instead of connecting two unrelated positions.
            history_pts = recent_contiguous_tail(uid, frame)
                    
            if len(history_pts) > 1:
                hx = [p[0] for p in history_pts]
                hy = [p[1] for p in history_pts]
                line, = ax.plot(hx, hy, "-", color=color, alpha=0.7, linewidth=2, zorder=2)
                dynamic_artists.append(line)
            
            # 2. Draw current user pos (Green circle)
            point, = ax.plot(curr_pos[0], curr_pos[1], "o", color="green", markersize=10, markeredgecolor="black", markeredgewidth=1, zorder=3)
            dynamic_artists.append(point)

        # 3. Set text states
        ax.set_title(f"Real Scenario - Step: {frame}/{max_steps}", fontsize=16)
        
        stats_str = f"Active Users: {len(users_at_frame)}"
        info_text.set_text(stats_str)
        
        return dynamic_artists + [info_text]

    print(f"Creating animation for {max_steps} frames...")
    ani = animation.FuncAnimation(fig, update, frames=max_steps + 1,
                                  init_func=init, blit=True, interval=200, repeat=False)

    plt.tight_layout()
    
    if save_path:
        print(f"Saving animation to {save_path}...")
        # Save as MP4 or GIF based on the extension
        writer = animation.FFMpegWriter(fps=5) if save_path.endswith('.mp4') else 'pillow'
        ani.save(save_path, writer=writer)
        print("Save completed!")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Real OSM ScalableMEC Experience")
    parser.add_argument('--scenario', type=str, default=None, help='Specific YAML scenario to load')
    parser.add_argument('--save', type=str, help='Path to save the animation (e.g., output.gif)', default=None)
    args = parser.parse_args()
    
    visualize_environment(scenario_file=args.scenario, save_path=args.save)
