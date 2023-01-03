import argparse

from maple.launchers.launcher_util import run_experiment
from maple.launchers.robosuite_launcher import experiment
import maple.util.hyperparameter as hyp
import collections

base_variant = dict(
    layer_size=256,
    replay_buffer_size=int(1E6),
    rollout_fn_kwargs=dict(
        terminals_all_false=True,
    ),
    algorithm_kwargs=dict(
        num_epochs=10000,
        num_expl_steps_per_train_loop=3000,
        num_eval_steps_per_epoch=3000,
        num_trains_per_train_loop=1000,
        min_num_steps_before_training=30000,
        max_path_length=150,
        batch_size=1024,
        eval_epoch_freq=10,
    ),
    trainer_kwargs=dict(
        discount=0.99,
        soft_target_tau=1e-3,
        target_update_period=1,
        policy_lr=3e-5,
        qf_lr=3e-5,
        reward_scale=1,
        use_automatic_entropy_tuning=True,
    ),
    ll_sac_variant=dict(
        high_init_ent=True,
    ),
    pamdp_variant=dict(
        one_hot_s=True,
        high_init_ent=True,
        one_hot_factor=0.50,
    ),
    env_variant=dict(
        robot_keys=['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 'robot0_gripper_qvel'],
        obj_keys=['object-state'],
        controller_type='OSC_POSITION_YAW',
        controller_config_update=dict(
            position_limits=[
                [-0.30, -0.30, 0.75],
                [0.15, 0.30, 1.15]
            ],
        ),
        env_kwargs=dict(
            ignore_done=True,
            reward_shaping=True,
            hard_reset=False,
            control_freq=10,
            camera_heights=512,
            camera_widths=512,
            table_offset=[-0.075, 0, 0.8],
            reward_scale=5.0,

            skill_config=dict(
                skills=['atomic', 'open', 'reach', 'grasp', 'push'],
                aff_penalty_fac=15.0,

                base_config=dict(
                    global_xyz_bounds=[
                        [-0.30, -0.30, 0.80],
                        [0.15, 0.30, 0.95]
                    ],
                    lift_height=0.95,
                    binary_gripper=True,

                    aff_threshold=0.06,
                    aff_type='dense',
                    aff_tanh_scaling=10.0,
                ),
                atomic_config=dict(
                    use_ori_params=True,
                ),
                reach_config=dict(
                    use_gripper_params=False,
                    local_xyz_scale=[0.0, 0.0, 0.06],
                    use_ori_params=False,
                    max_ac_calls=15,
                ),
                grasp_config=dict(
                    global_xyz_bounds=[
                        [-0.30, -0.30, 0.80],
                        [0.15, 0.30, 0.85]
                    ],
                    aff_threshold=0.03,

                    local_xyz_scale=[0.0, 0.0, 0.0],
                    use_ori_params=True,
                    max_ac_calls=20,
                    num_reach_steps=2,
                    num_grasp_steps=3,
                ),
                push_config=dict(
                    global_xyz_bounds=[
                        [-0.30, -0.30, 0.80],
                        [0.15, 0.30, 0.85]
                    ],
                    delta_xyz_scale=[0.25, 0.25, 0.05],

                    max_ac_calls=20,
                    use_ori_params=True,

                    aff_threshold=[0.12, 0.12, 0.04],
                ),
            ),
        ),
    ),
    save_video=True,
    save_video_period=100,
    dump_video_kwargs=dict(
        rows=1,
        columns=6,
        pad_length=5,
        pad_color=0,
    ),
)

env_params = dict(
    lift={
        'env_variant.env_type': ['Lift'],
    },
    door={
        'env_variant.env_type': ['Door'],
        'env_variant.controller_type': ['OSC_POSITION'],
        'env_variant.controller_config_update.position_limits': [[[-0.25, -0.25, 0.90], [0.05, 0.05, 1.20]]],
        'env_variant.env_kwargs.skill_config.base_config.global_xyz_bounds': [
            [[-0.25, -0.25, 0.90], [0.05, 0.05, 1.20]]],
        'env_variant.env_kwargs.skill_config.grasp_config.global_xyz_bounds': [
            [[-0.25, -0.25, 0.90], [0.05, 0.05, 1.20]]],
        'env_variant.env_kwargs.skill_config.push_config.global_xyz_bounds': [
            [[-0.25, -0.25, 0.90], [0.05, 0.05, 1.20]]],
        'env_variant.env_kwargs.skill_config.base_config.lift_height': [1.15],
        'env_variant.env_kwargs.skill_config.grasp_config.aff_threshold': [0.06],
        'env_variant.env_kwargs.skill_config.skills': [['atomic', 'grasp', 'reach_osc', 'push', 'open']],
    },
    pnp={
        'env_variant.env_type': ['PickPlaceCan'],
        'env_variant.env_kwargs.bin1_pos': [[0.0, -0.25, 0.8]],
        'env_variant.env_kwargs.bin2_pos': [[0.0, 0.28, 0.8]],
        'env_variant.controller_config_update.position_limits': [[[-0.15, -0.50, 0.75], [0.15, 0.50, 1.15]]],
        'env_variant.env_kwargs.skill_config.base_config.global_xyz_bounds': [
            [[-0.15, -0.50, 0.82], [0.15, 0.50, 1.02]]],
        'env_variant.env_kwargs.skill_config.grasp_config.global_xyz_bounds': [
            [[-0.15, -0.50, 0.82], [0.15, 0.50, 0.88]]],
        'env_variant.env_kwargs.skill_config.push_config.global_xyz_bounds': [
            [[-0.15, -0.50, 0.82], [0.15, 0.50, 0.88]]],
        'env_variant.env_kwargs.skill_config.base_config.lift_height': [1.0],
        'env_variant.env_kwargs.skill_config.reach_config.aff_threshold': [[0.15, 0.25, 0.06]],
    },
    wipe={
        'env_variant.env_type': ['Wipe'],
        'env_variant.obj_keys': [['robot0_contact-obs', 'object-state']],
        'algorithm_kwargs.max_path_length': [300],
        'env_variant.controller_type': ['OSC_POSITION'],
        'env_variant.controller_config_update.position_limits': [[[-0.10, -0.30, 0.75], [0.20, 0.30, 1.00]]],
        'env_variant.env_kwargs.table_offset':[[0.05, 0, 0.8]],
        'env_variant.env_kwargs.skill_config.base_config.global_xyz_bounds': [
            [[-0.10, -0.30, 0.75], [0.20, 0.30, 1.00]]],
        'env_variant.env_kwargs.skill_config.grasp_config.global_xyz_bounds': [
            [[-0.10, -0.30, 0.80], [0.20, 0.30, 0.85]]],
        'env_variant.env_kwargs.skill_config.push_config.global_xyz_bounds': [
            [[-0.10, -0.30, 0.80], [0.20, 0.30, 0.85]]],
        'env_variant.env_kwargs.skill_config.base_config.aff_threshold': [[0.15, 0.25, 0.03]],
        'env_variant.env_kwargs.skill_config.reach_config.aff_threshold': [[0.15, 0.25, 0.03]],
        'env_variant.env_kwargs.skill_config.grasp_config.aff_threshold': [[0.15, 0.25, 0.03]],
        'env_variant.env_kwargs.skill_config.push_config.aff_threshold': [[0.15, 0.25, 0.03]],
        'env_variant.env_kwargs.skill_config.skills': [['atomic', 'reach', 'push']],
    },
    stack={
        'env_variant.env_type': ['Stack'],
    },
    nut={
        'env_variant.env_type': ['NutAssemblyRound'],
        'env_variant.env_kwargs.skill_config.grasp_config.aff_threshold': [0.06],
    },
    cleanup={
        'env_variant.env_type': ['Cleanup'],
    },
    peg_ins={
        'env_variant.env_type': ['PegInHole'],
        'env_variant.controller_config_update.position_limits': [[[-0.30, -0.30, 0.75], [0.15, 0.30, 1.00]]],
        'env_variant.env_kwargs.skill_config.reach_config.aff_threshold': [0.06],
        'pamdp_variant.one_hot_factor': [0.375],
    },
)

def process_variant(variant):
    if args.debug:
        variant['algorithm_kwargs']['num_epochs'] = 3
        variant['algorithm_kwargs']['batch_size'] = 64
        steps = 50
        variant['algorithm_kwargs']['max_path_length'] = steps
        variant['algorithm_kwargs']['num_eval_steps_per_epoch'] = steps
        variant['algorithm_kwargs']['num_expl_steps_per_train_loop'] = steps
        variant['algorithm_kwargs']['min_num_steps_before_training'] = steps
        variant['algorithm_kwargs']['num_trains_per_train_loop'] = 50
        variant['replay_buffer_size'] = int(1E3)
        variant['dump_video_kwargs']['columns'] = 2

    if args.no_video:
        variant['save_video'] = False

    variant['exp_label'] = args.label

    return variant

def deep_update(source, overrides):
    """
    Copied from https://stackoverflow.com/questions/3232943/update-value-of-a-nested-dictionary-of-varying-depth
    """
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
    parser.add_argument('--no_video', action='store_true')
    parser.add_argument('--no_gpu', action='store_true')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--first_variant', action='store_true')
    parser.add_argument('--snapshot_gap', type=int, default=25)

    args = parser.parse_args()

    search_space = env_params[args.env]

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=base_variant,
    )
    for exp_id, variant in enumerate(sweeper.iterate_hyperparameters()):
        variant = process_variant(variant)

        run_experiment(
            experiment,
            exp_folder=args.env,
            exp_prefix=args.label,
            variant=variant,
            snapshot_mode='gap_and_last',
            snapshot_gap=args.snapshot_gap,
            exp_id=exp_id,
            use_gpu=(not args.no_gpu),
            gpu_id=args.gpu_id,
            mode='local',
            num_exps_per_instance=1,
        )

        if args.first_variant:
            exit()
