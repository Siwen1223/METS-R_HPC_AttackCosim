import carla
import matplotlib.pyplot as plt
from collections import defaultdict

def sample_segment_points(wp_start: carla.Waypoint, wp_end: carla.Waypoint, step=2.0, max_iter=2000):
    """
    从 wp_start 沿车道中心线采样点，直到接近 wp_end 为止。
    返回 [(x,y), ...]
    """
    pts = []
    wp = wp_start

    end_loc = wp_end.transform.location
    for _ in range(max_iter):
        loc = wp.transform.location
        pts.append((loc.x, loc.y))

        # 到达终点附近就停（避免无限走）
        if loc.distance(end_loc) < step * 1.5:
            pts.append((end_loc.x, end_loc.y))
            break

        nxt = wp.next(step)
        if not nxt:
            break
        wp = nxt[0]

        # 如果 road_id / lane_id 已经变了，说明跨段了，也停
        if wp.road_id != wp_start.road_id or wp.lane_id != wp_start.lane_id:
            break

    return pts

def draw_town05_road_ids(host="localhost", port=2000, town="Town05", step=2.0, show_lane_direction=False):
    client = carla.Client(host, port)
    client.set_timeout(20.0)

    # 1) 切换到 Town05
    world = client.load_world(town)
    carla_map = world.get_map()

    topology = carla_map.get_topology()

    # 2) 收集每个 road_id 的折线（按 lane 分开画）
    road_polylines = defaultdict(list)  # road_id -> list of polylines; each polyline is [(x,y),...]
    road_points_for_label = defaultdict(list)

    for wp_start, wp_end in topology:
        pts = sample_segment_points(wp_start, wp_end, step=step)
        if len(pts) < 2:
            continue

        rid = wp_start.road_id
        road_polylines[rid].append(pts)

        # 用于后面放 label：取这段中间点
        mid = pts[len(pts)//2]
        road_points_for_label[rid].append(mid)

    # 3) 绘图：先画所有折线，再标 road_id
    plt.figure(figsize=(12, 12))

    # 画路网（每条 polyline 一条线）
    for rid, polylines in road_polylines.items():
        for pts in polylines:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, linewidth=0.8)

            # 可选：画方向箭头（看车道方向）
            if show_lane_direction and len(pts) >= 2:
                x0, y0 = pts[len(pts)//2]
                x1, y1 = pts[len(pts)//2 + 1]
                plt.arrow(x0, y0, x1-x0, y1-y0, length_includes_head=True, head_width=1.2, head_length=2.0)

    # 标 road_id（每个 road_id 标一次：取所有 mid 点的平均）
    for rid, mids in road_points_for_label.items():
        if not mids:
            continue
        mx = sum(p[0] for p in mids) / len(mids)
        my = sum(p[1] for p in mids) / len(mids)
        plt.text(mx, my, str(rid), fontsize=9, ha="center", va="center")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(f"{town}: Road network with road_id labels (step={step}m)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == "__main__":
    #draw_town05_road_ids(town="Town05", step=2.0, show_lane_direction=False)
