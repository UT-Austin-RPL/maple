import numpy as np
import cv2
import os.path as osp
import math

def dump_video(
        env,
        policy,
        rollout_function,
        mode,
        epoch,
        rows=3,
        columns=6,
        pad_length=0,
        pad_color=255,
        do_timer=True,
        horizon=100,
        imsize=84,
        num_channels=3,
        rollout_fn_kwargs={},
):
    import os
    import skvideo.io
    import time
    from maple.core import logger

    assert mode in ['expl', 'eval']
    logdir = logger.get_snapshot_dir()
    if mode == 'expl':
        logdir = osp.join(logdir, 'vis_expl')
        if not os.path.exists(logdir):
            os.mkdir(logdir)

    frames = []
    H = imsize
    W = imsize
    N = rows * columns

    def addl_info_func(env, agent, o, a):
        skill = env.skill_controller.get_skill_name_from_action(a)
        info = dict(
            skill=skill,
        )
        return info

    rollout_actions = []
    num_ac_calls = []
    successes = []

    for i in range(N):
        start = time.time()
        path = rollout_function(
            env,
            policy,
            max_path_length=horizon,
            render=False,
            addl_info_func=addl_info_func,
            image_obs_in_info=True,
            **rollout_fn_kwargs
        )

        rollout_actions.append(path['actions'])
        num_ac_calls.append([info['num_ac_calls'] for info in path['env_infos']])
        successes.append([info.get('success', False) for info in path['env_infos']])

        sc = env.env.skill_controller
        skill_name_map = sc.get_full_skill_name_map()

        l = []
        for j in range(len(path['env_infos'])):
            imgs = path['env_infos'][j]['image_obs']
            skill = path['addl_infos'][j]['skill']

            skill_name = skill_name_map[skill]
            ac_str = skill_name

            success = successes[i][max(j-1, 0)]

            for img in imgs:
                img = np.flip(img, axis=0)
                img[-80:,:,:] = 235
                img = get_image(
                    img,
                    pad_length=pad_length,
                    pad_color=(0, 225, 0) if success else pad_color,
                    imsize=imsize,
                )
                if success:
                    ac_str = 'Success'
                img = annotate_image(
                    img,
                    text=ac_str,
                    imsize=imsize,
                    color=(0, 175, 0) if success else (0,0,0,),
                    loc='ll',
                )
                l.append(img)

            if success:
                break

        frames.append(l)

        if do_timer:
            print(i, time.time() - start)

    for i in range(len(frames)):
        last_img = frames[i][-1]
        for _ in range(horizon - len(frames[i])):
            frames[i].append(last_img)

    frames = np.array(frames, dtype=np.uint8)
    path_length = frames.size // (
            N * (H + 2 * pad_length) * (W + 2 * pad_length) * num_channels
    )
    frames = np.array(frames, dtype=np.uint8).reshape(
        (N, path_length, H + 2 * pad_length, W + 2 * pad_length, num_channels)
    )
    f1 = []
    for k1 in range(columns):
        f2 = []
        for k2 in range(rows):
            k = k1 * rows + k2
            f2.append(frames[k:k + 1, :, :, :, :].reshape(
                (path_length, H + 2 * pad_length, W + 2 * pad_length,
                 num_channels)
            ))
        f1.append(np.concatenate(f2, axis=1))
    outputdata = np.concatenate(f1, axis=2)

    filename = osp.join(
        logdir,
        'video_{mode}_{epoch}.mp4'.format(mode=mode, epoch=epoch)
    )
    import imageio
    with imageio.get_writer(filename, fps=24) as writer:
        for frame in outputdata:
            writer.append_data(frame)
    print("Saved video to ", filename)

    dump_skillmap(
        env,
        rollout_actions,
        successes,
        horizon,
        logdir, mode, epoch
    )


def dump_skillmap(
        env,
        rollout_actions,
        successes,
        horizon,
        logdir,
        mode,
        epoch,
):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    import matplotlib.colorbar

    num_rollouts = len(rollout_actions)
    horizon = np.max([len(rollout) for rollout in rollout_actions])
    horizon = math.ceil(horizon / 25) * 25 # round to next increment of 25
    skill_ids = -1 * np.ones((num_rollouts, horizon))

    sc = env.env.skill_controller
    skill_names, colors = sc.get_skill_names_and_colors()
    num_skills = len(skill_names)

    for i in range(num_rollouts):
        rollout = rollout_actions[i]
        rollout_len = len(rollout)
        for j in reversed(range(rollout_len)):
            ac = rollout[j]
            skill_id = sc.get_skill_id_from_action(ac)
            skill_ids[i][j] = skill_id
            if j-1 >= 0 and successes[i][j-1]:
                skill_ids[i][j] += num_skills

    colors.insert(0, "white")
    colors_rgba = []
    colors_faint_rgba = []
    for (i, c) in enumerate(colors):
        rgba = matplotlib.colors.to_rgba(c)
        colors_rgba.append(rgba)
        rgba = list(rgba)
        rgba[-1] = 0.30
        if i != 0:
            colors_faint_rgba.append(rgba)
    cmap = ListedColormap(colors_rgba + colors_faint_rgba)
    cmap_vis = ListedColormap(colors_rgba)
    plt.figure( figsize = (15 * math.ceil(horizon / 100) ,3))
    plt.pcolormesh(skill_ids, edgecolors='w', linewidth=1, cmap=cmap, vmin=-1.5, vmax=num_skills*2 - 0.5)
    plt.yticks([])
    ax = plt.gca()
    ax.set_aspect('equal')
    cax, _ = matplotlib.colorbar.make_axes(plt.gca(), location = 'right')
    ticks = 1/(num_skills+1) * (np.arange(1, num_skills + 1) + 0.5)
    cbar = matplotlib.colorbar.ColorbarBase(
        cax,
        cmap=cmap_vis,
        ticks=ticks
    )
    cbar.ax.set_yticklabels(skill_names)
    filename = osp.join(
        logdir,
        'cmap_{mode}_{epoch}.png'.format(mode=mode, epoch=epoch)
    )
    plt.savefig(filename, bbox_inches='tight')


def get_image(obs, imsize=84, pad_length=1, pad_color=255):
    if len(obs.shape) == 1:
        obs = obs.reshape(-1, imsize, imsize).transpose()
    img = obs

    # img = np.uint8(255 * img)

    if pad_length > 0:
        img = add_border(img, pad_length, pad_color, imsize=imsize)
    return img


def add_border(img, pad_length, pad_color, imsize=84):
    H = imsize
    W = imsize
    img = img.reshape((imsize, imsize, -1))
    img2 = np.ones((H + 2 * pad_length, W + 2 * pad_length, img.shape[2]),
                   dtype=np.uint8) * np.array(pad_color, dtype=np.uint8)
    img2[pad_length:-pad_length, pad_length:-pad_length, :] = img
    return img2

def annotate_image(img, text, imsize=84, color=(0, 0, 255), loc='ll'):
    img = img.copy()

    fontScale = 0.30 / 84 * imsize
    if imsize == 1024:
        thickness = 10
        fontScale *= 1.5
    elif imsize == 512:
        thickness = 5
    elif imsize == 84:
        thickness = 1
    else:
        thickness = 1

    if loc == 'll':
        org = (30, imsize - 10)
    elif loc == 'lr':
        if imsize == 1024:
            org = (imsize - 175, imsize - 40)
        else:
            org = (imsize - 100, imsize - 10)
    else:
        raise ValueError
    fontFace = 0

    textsize = cv2.getTextSize(text, fontFace, fontScale, thickness)[0]
    textX = int((img.shape[1] - textsize[0]) / 2)
    cv2.putText(img=img, text=text, org=(textX, org[1]), fontFace=fontFace, fontScale=fontScale,
                color=color, thickness=thickness)
    return img

def normalize_image(image):
    assert image.dtype == np.uint8
    return np.float32(image) / 255.0

def unormalize_image(image):
    assert image.dtype != np.uint8
    return np.uint8(image * 255.0)
