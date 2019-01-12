import json
import click
from tqdm import trange
from nbody.run_nbody import run


@click.command()
@click.option("--config", default="config.json", help="Config")
def main(config="config.json"):

    with open(config) as f:
        simulation_params = json.load(f)
    gpu_simulation_params = simulation_params.copy()
    simulation_params["gpu"] = False
    gpu_simulation_params["gpu"] = True

    file_path_template = "/mnt/hdd/data/{}/data{{0:08d}}.h5"
    for power in trange(3, 16, 3):
        N = int(2 ** power)
        simulation_params["N"] = N
        gpu_simulation_params["N"] = N
        simulation_params["file_path"] = file_path_template.format(f"cpu{N}")
        gpu_simulation_params["file_path"] = file_path_template.format(f"gpu{N}")
        run(**gpu_simulation_params, save_dense_files=False)
        run(**simulation_params, save_dense_files=False)


if __name__ == "__main__":
    main()
