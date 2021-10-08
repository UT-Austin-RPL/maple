from maple.launchers.launcher_util import run_experiment
from maple.launchers.robosuite_launcher import experiment
import maple.util.hyperparameter as hyp
import os.path as osp
import argparse
import json
import collections
import copy

from maple.launchers.conf import LOCAL_LOG_DIR

base_variant = dict(
    algorithm_kwargs=dict(
        eval_only=True,
        num_epochs=5000,
        eval_epoch_freq=100,
    ),
    replay_buffer_size=int(1E2),
    vis_expl=False,
    dump_video_kwargs=dict(
        rows=1,
        columns=6,
        pad_length=5,
        pad_color=0,
    ),
    num_eval_rollouts=50,

    # ckpt_epoch=100, #### uncomment if you want to evaluate a specific epoch ckeckpoint only ###
)

env_params = dict(
    lift={
        'ckpt_path': [
            ### Add paths here ###
        ],
    },
    door={
        'ckpt_path': [
            ### Add paths here ###
        ],
    },
    pnp={
        'ckpt_path': [
            ### Add paths here ###
        ],
    },
    wipe={
        'ckpt_path': [
            ### Add paths here ###
        ],
    },
    stack={
        'ckpt_path': [
            ### Add paths here ###
        ],
    },
    nut_round={
        'ckpt_path': [
            ### Add paths here ###
        ],
    },
    cleanup={
        'ckpt_path': [
            ### Add paths here ###
        ],
    },
    peg_ins={
        'ckpt_path': [
            ### Add paths here ###
        ],
    },
)

def process_variant(eval_variant):
    ckpt_path = eval_variant['ckpt_path']
    json_path = osp.join(LOCAL_LOG_DIR, ckpt_path, 'variant.json')
    with open(json_path) as f:
        ckpt_variant = json.load(f)
    deep_update(ckpt_variant, eval_variant)
    variant = copy.deepcopy(ckpt_variant)

    if args.debug:
        mpl = variant['algorithm_kwargs']['max_path_length']
        variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = mpl * 3
        variant['dump_video_kwargs']['rows'] = 1
        variant['dump_video_kwargs']['columns'] = 2
    else:
        mpl = variant['algorithm_kwargs']['max_path_length']
        variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = mpl * variant['num_eval_rollouts']

    variant['save_video_period'] = variant['algorithm_kwargs']['eval_epoch_freq']

    if args.no_video:
        variant['save_video'] = False

    variant['exp_label'] = args.label
    return variant

def deep_update(source, overrides):
    '''
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    Copied from: https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    '''
    for key, value in overrides.items():
        if isinstance(value, collections.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source

if __name__ == "__main__":
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str)
    parser.add_argument('--label', type=str, default='test')
    parser.add_argument('--num_seeds', type=int, default=1)
    parser.add_argument('--no_video', action='store_true')
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--first_variant', action='store_true')
    args = parser.parse_args()

    search_space = env_params[args.env]
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=base_variant,
    )
    for exp_id, eval_variant in enumerate(sweeper.iterate_hyperparameters()):
        variant = process_variant(eval_variant)

        run_experiment(
            experiment,
            exp_folder=args.env,
            exp_prefix=args.label,
            variant=variant,
            snapshot_mode='gap_and_last',
            snapshot_gap=200,
            exp_id=exp_id,
            use_gpu=(not args.no_gpu),
            gpu_id=args.gpu_id,
        )

        if args.first_variant:
            exit()