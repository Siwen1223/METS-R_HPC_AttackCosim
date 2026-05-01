import sys
import os
import argparse
import time
import xml.etree.ElementTree as ET

# This script lives in tutorials/, but it needs to import from the repo's
# top-level `clients/` and `utils/` packages and resolve relative paths such as
# `configs/...`, `data/...`, and `docker/` against the repo root. Make that work
# regardless of where the script is invoked from.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
os.chdir(_REPO_ROOT)

from clients.CoSimClient import CoSimClient
from utils.util import read_run_config, prepare_sim_dirs, run_simulations, run_simulations_in_background, run_simulation_in_docker 
from utils.carla_util import open_carla

# use case: python tutorials/cosim_example.py -r configs/run_cosim_CARLAT5.json -v
def get_arguments(argv):
    parser = argparse.ArgumentParser(description='METS-R simulation')
    parser.add_argument('-r','--run_config', default='configs/run_cosim_CARLAT5.json',
                        help='the folder that contains all the input data')
    parser.add_argument('-a', '--display_all', action='store_true', default=False, help='display all vehicles')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='verbose mode')
    args = parser.parse_args(argv)

    config = read_run_config(args.run_config)
    config.display_all = args.display_all
    config.verbose = args.verbose

    return config


def get_all_roads(network_file):
    """Return all road IDs from a SUMO net.xml and its paired xodr file.

    metsr_road  - all non-internal SUMO edge IDs (strings, e.g. "-47", "0").
                  Internal junction connector edges (id starts with ':') are excluded.
    carla_road  - all road IDs from the OpenDRIVE xodr file (ints), covering both
                  regular roads (junction="-1") and junction connector roads.
    """
    # --- SUMO net.xml → metsr_road ---
    tree = ET.parse(network_file)
    root = tree.getroot()
    metsr_roads = [
        edge.get('id')
        for edge in root.findall('edge')
        if not edge.get('id', '').startswith(':')
    ]

    # --- OpenDRIVE xodr → carla_road ---
    xodr_file = network_file.replace('.net.xml', '.xodr')
    carla_roads = []
    if os.path.exists(xodr_file):
        xodr_tree = ET.parse(xodr_file)
        xodr_root = xodr_tree.getroot()
        carla_roads = [int(road.get('id')) for road in xodr_root.findall('road')]
    else:
        print(f"Warning: xodr file not found at {xodr_file}, carla_road will be empty.")

    return metsr_roads, carla_roads


if __name__ == '__main__':
    config = get_arguments(sys.argv[1:])
    os.chdir("docker")
    os.system("docker-compose up -d")
    os.chdir("..")

    # Prepare simulation directories
    dest_data_dirs = prepare_sim_dirs(config)

    # Derive co-sim road lists directly from the network files so the config
    # stays correct for any map, not just the hardcoded Town05 subset.
    metsr_roads, carla_roads = get_all_roads(config.network_file)
    to_add_config = {"metsr_road": metsr_roads, "carla_road": carla_roads}


    for key, value in to_add_config.items():
        setattr(config, key, value)

    # run_co_simulation
    carla_client, carla_tm = open_carla(config)

    # Launch the simulations
    container_ids = run_simulation_in_docker(config)

    client = CoSimClient(config, carla_client, carla_tm)
    client.run()
