import argparse
import os

import torch
import numpy as np
import pandas as pd

from sklearn.manifold import TSNE

import plotly.express as px

DEFAULT_SCHEDULE = [
    (0, 1500, 500),
    (1500, 1860, 100),
    (1860, 2001, 20),
]


def plot_tsne(
        samples: list,
        sent_lens: list,
        start_timestep: int = None,
        end_timestep: int = None,
        step_size: int = None,
        schedule: list = None,
        output_dir: str = '',
        rank: int = None,
        batch_id: int = None,
):

    samples = [inst.cpu() for inst in samples]

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    if step_size == end_timestep == start_timestep is None:
        schedule = DEFAULT_SCHEDULE

    timesteps = len(samples)
    batch_size = len(samples[0])
    sequence_length = len(samples[0][0])

    for instance_id in range(batch_size):
        instance_samples = []

        # get timestamp samples
        if schedule:
            instance_samples = _scheduled_tsne_sampling(samples, schedule, timesteps, instance_id)

        else:
            for timestep in range(start_timestep, end_timestep, step_size):
                if timestep == timesteps:
                    timestep -= 1
                instance_samples.append((samples[timestep][instance_id], timestep))

        sample_dfs = [(pd.DataFrame(s), ts) for (s, ts) in instance_samples]

        labels = [
                     'Sentence' for _ in range(sent_lens[instance_id])
                 ] + [
                     'Scanpath' for _ in range(sequence_length - sent_lens[instance_id])
                 ]

        for df, ts in sample_dfs:
            df['label'] = labels

        for df, ts in sample_dfs:
            features = df.loc[:, df.columns != 'label']

            tsne = TSNE(
                n_components=2,
                random_state=0,
                perplexity=5,
                init="pca",
                n_iter=1000,
                early_exaggeration=30,
            )
            projections = tsne.fit_transform(features)

            title = fr'$sent-id = {instance_id},t = {timesteps - ts}$'

            fig = px.scatter(
                projections, x=0, y=1,
                color=df.label,
                width=600, height=300, title=title,
                labels={'0': 't-SNE component 1',
                        '1': 't-SNE component 2',
                        'color': 'Colors'}
            )

            # fig.show()
            filename = f'sp_tSNE_rank{rank}_batch{batch_id}_instance{instance_id}_t{ts}.png'
            file_path = os.path.join(output_dir, filename)
            fig.write_image(file_path)


def _scheduled_tsne_sampling(output_samples: list, schedule: list, max_ts: int, instance: int):
    samples = []

    for ts in sorted(schedule):
        try:
            for timestep in range(*ts):
                if timestep == max_ts:
                    timestep -= 1
                samples.append((output_samples[timestep][instance], timestep))

        except:
            raise ValueError('Something with your schedule seems to be wrong, please check it.')

    return samples
