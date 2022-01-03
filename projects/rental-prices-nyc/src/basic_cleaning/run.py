#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()

    logger.info(f'Download {args.input_artifact} artifact')
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)
    logger.info(f'Retrieved data has {df.shape[0]} records and {df.shape[1]} '
                'columns')

    # Filter on price
    df = df.loc[df['price'].between(args.min_price, args.max_price)]
    logger.info(f'Filter out records with price not between {args.min_price} '
                f'and {args.max_price}')
    logger.info(f'{df.shape[0]} records still available')

    # Filter on proper geo location
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5,
                                                                           41.2)
    df = df[idx].copy()

    # Output data
    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description
    )
    logger.info(f'Load filtered data to {args.output_artifact}')
    df.to_csv('./clean_sample.csv', index=False)
    artifact.add_file('./clean_sample.csv')
    run.log_artifact(artifact)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help='Input artifact',
        required=True
    )

    parser.add_argument(
        "--output_artifact", 
        type=str,
        help='Output artifact',
        required=True
    )

    parser.add_argument(
        "--output_type", 
        type=str,
        help="Output artifact's type",
        required=True
    )

    parser.add_argument(
        "--output_description", 
        type=str,
        help="Output artifact's description",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help='Minimum price acceptable',
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help='Maximum price acceptable',
        required=True
    )

    args = parser.parse_args()

    go(args)
