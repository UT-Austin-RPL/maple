import os
import argparse
import pathlib

def create_sbatch_script(args, use_variants=True):
    # Create a new directory path if it doesn't exist and create a new filename that we will write to
    exp_dir = args.exp_dir
    sbatch_dir = os.path.join(exp_dir, "sbatch")
    new_sbatch_fpath = os.path.join(sbatch_dir, "{}.sbatch".format(args.job_name))
    if not os.path.isdir(sbatch_dir):
        os.mkdir(sbatch_dir)

    if use_variants:
        base_variant = os.path.join(exp_dir, "variants", args.env, "base.json")
        variant_update = os.path.join(exp_dir, "variants", args.env, "{}.json".format(args.config))
        assert os.path.exists(base_variant) and os.path.exists(variant_update)

    command = ""
    for i in range(args.num_seeds):
        # Compose main command to be executed in script
        python_script = args.python_script
        line_command = "{{\nsleep {}\n".format(30*i)
        line_command += "python {python_script} --env {env} --label {label}".format(
            python_script=python_script,
            env=args.env,
            label=args.label,
        )
        if use_variants:
            line_command = "{line_command} --base_variant {base_variant} --variant_update {variant_update}".format(
                line_command=line_command,
                base_variant=base_variant,
                variant_update=variant_update,
            )
        if args.no_gpu:
            line_command += " --no_gpu"

        command += "{line_command}\n}} & \n".format(line_command=line_command)
    command += "wait"

    if args.partition in ["titans", "dgx"]:
        log_dir = "/scratch/cluster/soroush/logs"
    elif args.partition in ["svl", "tibet", "napoli-gpu"]:
        log_dir = "/cvgl2/u/soroush/logs"
    else:
        raise ValueError

    if args.exclude is None:
        if args.partition == "titans":
            args.exclude = "titan-5,titan-12"
        else:
            args.exclude = ""

    # Define a dict to map expected fill-ins with replacement values
    fill_ins = {
        "{{PARTITION}}": args.partition,
        "{{EXCLUDE}}": args.exclude,
        "{{NUM_GPU}}": 0 if args.no_gpu else 1,
        "{{NUM_CPU}}": args.num_seeds,
        "{{MEM}}": args.mem * args.num_seeds,
        "{{JOB_NAME}}": args.job_name,
        "{{LOG_DIR}}": log_dir,
        "{{HOURS}}": args.max_hours,
        "{{CMD}}": command,
        "{{CONDA_ENV}}": args.conda_env,
    }

    # Open the template file
    with open(args.slurm_template) as template:
        # Open the new sbatch file
        print(new_sbatch_fpath)
        with open(new_sbatch_fpath, 'w+') as new_file:
            # Loop through template and write to this new file
            for line in template:
                wrote = False
                # Check for various cases
                for k, v in fill_ins.items():
                    # If the key is found in the line, replace it with its value and pop it from the dict
                    if k in line:
                        new_file.write(line.replace(k, str(v)))
                        wrote = True
                        break
                # Otherwise, we just write the line from the template directly
                if not wrote:
                    new_file.write(line)

    # Execute this file!
    # TODO: Fix! (Permission denied error)
    #os.system(new_sbatch_fpath)

if __name__ == "__main__":
    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', type=str, default='can')
    parser.add_argument('--config', type=str, default='base')
    parser.add_argument('--label', type=str, default=None)
    parser.add_argument('--job_name', type=str, default=None)

    parser.add_argument('--num_seeds', type=int, default=4)
    parser.add_argument('--no_video', action='store_true')
    parser.add_argument('--no_gpu', action='store_true')

    parser.add_argument('--mem', type=int, default=9)
    parser.add_argument('--max_hours', type=int, default=504) #168
    parser.add_argument('--partition', type=str, default="titans")
    parser.add_argument('--exclude', type=str, default=None)

    args = parser.parse_args()

    if args.label is None:
        args.label = args.config

    if args.job_name is None:
        args.job_name = "rl_{}_{}".format(args.env, args.label)

    if args.exclude is None:
        if args.partition == "titans":
            args.exclude = "titan-5,titan-12"
        else:
            args.exclude = ""

    create_sbatch_script(args)
