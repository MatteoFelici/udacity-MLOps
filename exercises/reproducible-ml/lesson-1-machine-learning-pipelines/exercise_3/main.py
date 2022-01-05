import mlflow
import os
import hydra
from omegaconf import DictConfig


# This automatically reads in the configuration
@hydra.main(config_name="config")
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    _ = mlflow.run(
        os.path.join(root_path, "download_data"),
        parameters={
            "file_url": config["data"]["file_url"],
            "artifact_name": "iris.csv",
            "artifact_type": "raw_data",
            "artifact_description": "Input data",
        },
    )

    _ = mlflow.run(
        os.path.join(root_path, "process_data"),
        parameters={
            "input_artifact": "iris.csv:latest",
            "artifact_name": "cleaned_data.csv",
            "artifact_type": "clean_data",
            "artifact_description": "Cleaned data",
        },
    )


if __name__ == "__main__":
    go()
