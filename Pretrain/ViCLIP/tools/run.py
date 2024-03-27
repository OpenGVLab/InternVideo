import argparse
import os
import socket

from utils import has_slurm, random_port, runcmd

EXP_DIR_ENV_NAME = "VL_EXP_DIR"

# if key in hostname; apply the args in value to slurm.
DEFAULT_SLURM_ARGS = dict(login="-p gpu --mem=240GB -c 64 -t 2-00:00:00")


def get_default_slurm_args():
    """get the slurm args for different cluster.
    Returns: TODO

    """
    hostname = socket.gethostname()
    for k, v in DEFAULT_SLURM_ARGS.items():
        if k in hostname:
            return v
    return ""


def parse_args():
    parser = argparse.ArgumentParser()

    # slurm
    parser.add_argument("--slurm_args", type=str, default="", help="args for slurm.")
    parser.add_argument(
        "--no_slurm",
        action="store_true",
        help="If specified, will launch job without slurm",
    )
    parser.add_argument("--jobname", type=str, required=True, help="experiment name")
    parser.add_argument(
        "--dep_jobname", type=str, default="impossible_jobname", help="the dependent job name"
    )
    parser.add_argument("--nnodes", "-n", type=int, default=1, help="the number of nodes")
    parser.add_argument(
        "--ngpus", "-g", type=int, default=1, help="the number of gpus per nodes"
    )

    #
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="one of: pretrain, retrieval, retrieval_mc, vqa.",
    )
    parser.add_argument("--config", type=str, required=True, help="config file name.")
    parser.add_argument("--model_args", type=str, default="", help="args for model")

    args = parser.parse_args()
    return args


def get_output_dir(args):
    """get the output_dir"""
    return os.path.join(os.environ[EXP_DIR_ENV_NAME], args.jobname)


def prepare(args: argparse.Namespace):
    """prepare for job submission

    Args:
        args (dict): The arguments.

    Returns: The path to the copied source code.

    """

    output_dir = get_output_dir(args)
    code_dir = os.path.join(output_dir, "code")
    project_dirname = os.path.basename(os.getcwd())

    # check output_dir exist
    if os.path.isdir(output_dir):
        # if using slurm
        if has_slurm() and not args.no_slurm:
            raise ValueError(f"output_dir {output_dir} already exist. Exit.")
    else:
        os.mkdir(output_dir)
        # copy code
        cmd = f"cd ..; rsync -ar {project_dirname} {code_dir} --exclude='*.out'"
        print(cmd)
        runcmd(cmd)
    return os.path.join(code_dir, project_dirname)


def submit_job(args: argparse.Namespace):
    """TODO: Docstring for build_job_script.

    Args:
        args (argparse.Namespace): The commandline args.

    Returns: str. The script to run.

    """
    output_dir = get_output_dir(args)
    # copy code
    code_dir = prepare(args)

    # enter in the backup code
    master_port = os.environ.get("MASTER_PORT", random_port())
    init_cmd = f" cd {code_dir}; export MASTER_PORT={master_port}; "

    if has_slurm() and not args.no_slurm:
        # prepare slurm args.
        mode = "slurm"
        default_slurm_args = get_default_slurm_args()
        bin = (
            f" sbatch --output {output_dir}/%j.out --error {output_dir}/%j.out"
            f" {default_slurm_args}"
            f" {args.slurm_args} --job-name={args.jobname} --nodes {args.nnodes} "
            f" --ntasks {args.nnodes} "
            f" --gpus-per-node={args.ngpus} "
            f" --dependency=$(squeue --noheader --format %i --name {args.dep_jobname}) "
        )
    else:
        mode = "local"
        bin = "bash "

    # build job cmd
    job_cmd = (
        f" tasks/{args.task}.py"
        f" {args.config}"
        f" output_dir {output_dir}"
        f" {args.model_args}"
    )

    cmd = (
        f" {init_cmd} {bin} "
        f" tools/submit.sh "
        f" {mode} {args.nnodes} {args.ngpus} {job_cmd} "
    )

    with open(os.path.join(output_dir, "cmd.txt"), "w") as f:
        f.write(cmd)

    print(cmd)
    runcmd(cmd)


if __name__ == "__main__":
    args = parse_args()
    submit_job(args)
