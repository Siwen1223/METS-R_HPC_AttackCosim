from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.util import read_run_config


def _orig_roads_from_value(value):
    roads = set()
    for token in value.split():
        road_id = token.split("_", 1)[0]
        if road_id:
            roads.add(int(road_id))
    return roads


def derive_carla_road_from_metsr(metsr_road_ids, config_path):
    """
    Derive the CARLA road_id list corresponding to a set of METS-R road IDs.
    It reads the SUMO/METS-R network XML from the run config, collects road IDs
    referenced by lane origId fields on the selected edges, and also collects
    connector road IDs from connection origId fields between every selected road pair.
    """
    config = read_run_config(config_path)
    net_path = (ROOT_DIR / config.network_file).resolve()
    root = ET.parse(net_path).getroot()

    metsr_road_ids = [str(road_id) for road_id in metsr_road_ids]
    metsr_road_set = set(metsr_road_ids)

    edges = {edge.get("id"): edge for edge in root.findall("edge") if edge.get("id") is not None}
    carla_roads = set()

    for road_id in metsr_road_ids:
        edge = edges.get(road_id)
        if edge is None:
            continue
        for lane in edge.findall("lane"):
            for param in lane.findall("param"):
                if param.get("key") == "origId" and param.get("value"):
                    carla_roads.update(_orig_roads_from_value(param.get("value")))

    for connection in root.findall("connection"):
        from_road = connection.get("from")
        to_road = connection.get("to")
        if from_road not in metsr_road_set or to_road not in metsr_road_set:
            continue
        for param in connection.findall("param"):
            if param.get("key") == "origId" and param.get("value"):
                carla_roads.update(_orig_roads_from_value(param.get("value")))

    return sorted(carla_roads)


if __name__ == "__main__":
    metsr_roads = ["-0", "-1", "0", "1", "-18", "40", "17", "-47"]
    carla_roads = derive_carla_road_from_metsr(
        metsr_road_ids=metsr_roads,
        config_path="configs/run_cosim_CARLAT5.json",
    )
    print(carla_roads)
