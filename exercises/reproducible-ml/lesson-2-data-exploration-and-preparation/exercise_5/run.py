#!/usr/bin/env python
import argparse
import logging
import pandas as pd
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(project="exercise_5", job_type="process_data")

    logger.info(f"Retrieve {args.input_artifact} artifact from W&B")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()

    logger.info("Read input data")
    df = pd.read_parquet(artifact_path)
    logger.info(f"{df.shape[0]} records and {df.shape[1]} columns")

    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Drop duplicates - now {df.shape[0]} records")

    logger.info("Add text_feature feature")
    df["title"].fillna(value="", inplace=True)
    df["song_name"].fillna(value="", inplace=True)
    df["text_feature"] = df["title"] + " " + df["song_name"]

    logger.info(
        "Define an output artifact with:\n"
        f"- name {args.artifact_name}\n"
        f"- type {args.artifact_type}\n"
        f"- description {args.artifact_description}"
    )
    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )

    logger.info("Load output data to artifact")
    df.to_csv("preprocessed_data.csv", index=False)
    artifact.add_file("preprocessed_data.csv")

    run.log_artifact(artifact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess a dataset",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Fully-qualified name for the input artifact",
        required=True,
    )

    parser.add_argument(
        "--artifact_name", type=str, help="Name for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_type", type=str, help="Type for the artifact", required=True
    )

    parser.add_argument(
        "--artifact_description",
        type=str,
        help="Description for the artifact",
        required=True,
    )

    args = parser.parse_args()

    go(args)
