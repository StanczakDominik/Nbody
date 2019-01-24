import datetime
import json
import click
from tqdm import trange
from nbody.run_nbody import run

MAX_POWER = 4


@click.command()
@click.option("--config", default="config.json", help="Config")
def main(config="config.json"):

    with open(config) as f:
        simulation_params = json.load(f)
    gpu_simulation_params = simulation_params.copy()
    simulation_params["gpu"] = False
    gpu_simulation_params["gpu"] = True

    file_path_template = "/mnt/hdd/data/{}/data{{0:08d}}.h5"
    for i in range(1):
        for power in trange(3, MAX_POWER, 3):
            N = int(2 ** power)
            simulation_params["N"] = N
            gpu_simulation_params["N"] = N
            datestring = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            simulation_params["file_path"] = file_path_template.format(
                f"cpu{N}_{datestring}"
            )
            gpu_simulation_params["file_path"] = file_path_template.format(
                f"gpu{N}_{datestring}"
            )
            run(**gpu_simulation_params, save_dense_files=False)

        for power in trange(3, MAX_POWER, 3):
            N = int(2 ** power)
            simulation_params["N"] = N
            gpu_simulation_params["N"] = N
            datestring = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            simulation_params["file_path"] = file_path_template.format(
                f"cpu{N}_{datestring}"
            )
            gpu_simulation_params["file_path"] = file_path_template.format(
                f"gpu{N}_{datestring}"
            )
            run(**simulation_params, save_dense_files=False)


if __name__ == "__main__":
    main()
