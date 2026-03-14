import carla
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def sample_segment_points(wp_start: carla.Waypoint, wp_end: carla.Waypoint, step=2.0, max_iter=2000):
    pts = []
    wp = wp_start
    end_loc = wp_end.transform.location

    for _ in range(max_iter):
        loc = wp.transform.location
        pts.append((loc.x, loc.y))

        if loc.distance(end_loc) < step * 1.5:
            pts.append((end_loc.x, end_loc.y))
            break

        nxt = wp.next(step)
        if not nxt:
            break
        wp = nxt[0]

        if wp.road_id != wp_start.road_id or wp.lane_id != wp_start.lane_id:
            break

    return pts


def draw_town_road_ids(
    host="localhost",
    port=2000,
    town="Town05",
    step=2.0,
    show_lane_direction=False
):
    client = carla.Client(host, port)
    client.set_timeout(20.0)

    world = client.load_world(town)
    carla_map = world.get_map()
    topology = carla_map.get_topology()

    road_polylines = defaultdict(list)
    road_points_for_label = defaultdict(list)

    for wp_start, wp_end in topology:
        pts = sample_segment_points(wp_start, wp_end, step=step)
        if len(pts) < 2:
            continue

        rid = wp_start.road_id
        road_polylines[rid].append(pts)
        road_points_for_label[rid].append(pts[len(pts)//2])

    # ===== asign a color for every road_id  =====
    road_ids = sorted(road_polylines.keys())
    cmap = cm.get_cmap("tab20", len(road_ids))
    road_color = {rid: cmap(i) for i, rid in enumerate(road_ids)}

    plt.figure(figsize=(12, 12))

    # PLot the road network
    for rid, polylines in road_polylines.items():
        color = road_color[rid]
        for pts in polylines:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, linewidth=1.2, color=color)

            if show_lane_direction and len(pts) >= 2:
                x0, y0 = pts[len(pts)//2]
                x1, y1 = pts[len(pts)//2 + 1]
                plt.arrow(
                    x0, y0, x1-x0, y1-y0,
                    color=color,
                    length_includes_head=True,
                    head_width=1.2,
                    head_length=2.0
                )

    # Label road_id
    for rid, mids in road_points_for_label.items():
        mx = sum(p[0] for p in mids) / len(mids)
        my = sum(p[1] for p in mids) / len(mids)
        plt.text(
            mx, my, str(rid),
            fontsize=9,
            color=road_color[rid],
            ha="center",
            va="center",
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.6,
                pad=0.5
            )
        )

    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("off")
    plt.title(f"{town}: Road network with road_id (color-matched)")
    plt.show()


if __name__ == "__main__":
    draw_town_road_ids(town="Town05", step=2.0)
