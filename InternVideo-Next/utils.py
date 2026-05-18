import io
import os
import math
import time
import json
from collections import defaultdict, deque
import datetime
import numpy as np
from timm.utils import get_state_dict
from torch.utils.data._utils.collate import default_collate
from pathlib import Path
import subprocess
import torch
import torch.distributed as dist
from torch._six import inf
import random

from tensorboardX import SummaryWriter

import fnmatch
try:
    from petrel_client.client import Client
    has_client = True
    client = Client('~/petreloss.conf')
except ImportError:
    has_client = False
    client = None


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f} (max: {max:.4f})')
        data_time = SmoothedValue(fmt='{avg:.4f} (max: {max:.4f})')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

    def log_every_joint(self, video_loader, image_loader, print_freq, header=None, image_num_ratio=1.0):
        # prepare random squeue
        total_len = int(len(video_loader) + len(image_loader) * image_num_ratio)
        random_sequence = np.arange(total_len)
        np.random.shuffle(random_sequence)
        loader_list = [iter(video_loader), iter(image_loader)]
        # prepare print template
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f} (max: {max:.4f})')
        data_time = SmoothedValue(fmt='{avg:.4f} (max: {max:.4f})')
        space_fmt = ':' + str(len(str(total_len))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0

        for i, random_num in enumerate(random_sequence):
            # randomly selct image or video
            if random_num < len(video_loader):
                loader_idx = 0
                use_image = False
                mark = '<<VIDEO BATCH>>\t'
            else:
                loader_idx = 1
                use_image = True
                mark = '<<IMAGE BATCH>>\t'
            data_time.update(time.time() - end)
            yield (next(loader_list[loader_idx]), use_image)
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == total_len - 1:
                eta_seconds = iter_time.global_avg * (total_len - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(mark, log_msg.format(
                        i, total_len, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(mark, log_msg.format(
                        i, total_len, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / total_len))


class TensorboardLogger(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(logdir=log_dir)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head='scalar', step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.writer.add_scalar(head + "/" + k, v, self.step if step is None else step)

    def flush(self):
        self.writer.flush()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    

def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def get_ceph_path(ckpt_path, ceph_args):
    sub_path = str(ckpt_path).split(ceph_args['ckpt_path_split'])[-1]
    ceph_ckpt_path = os.path.join(ceph_args['ceph_checkpoint_prefix'], sub_path)
    return sub_path, ceph_ckpt_path

def save_on_master(obj, ckpt_path, ceph_args):
    if is_main_process():
        if ceph_args['use_ceph_checkpoint']:
            assert has_client == True, "petrel_client is not installed!!!"
            _, ceph_ckpt_path = get_ceph_path(ckpt_path, ceph_args)
            with io.BytesIO() as f:
                torch.save(obj, f)
                client.put(ceph_ckpt_path, f.getvalue())
        else:
            torch.save(obj, ckpt_path)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = int(os.environ['SLURM_LOCALID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        os.environ['RANK'] = str(args.rank)
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['WORLD_SIZE'] = str(args.world_size)

        node_list = os.environ['SLURM_NODELIST']
        addr = subprocess.getoutput(
            f'scontrol show hostname {node_list} | head -n1')
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = addr
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    # assert torch.distributed.is_initialized()
    setup_for_distributed(args.rank == 0)


def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0,
                     start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = int(warmup_epochs * niter_per_ep)
    if warmup_steps > 0:
        warmup_iters = warmup_steps
    print("Set warmup steps = %d" % warmup_iters)
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters])

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs * niter_per_ep
    return schedule


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, model_name=None, ceph_args={'use_ceph_checkpoint': False}):
    output_dir = Path(args.output_dir)
    if model_name is None:
        model_name = str(epoch)
    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % model_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path, ceph_args=ceph_args)
    else:
        client_state = {'epoch': epoch}
        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)

        if ceph_args['use_ceph_checkpoint']:
            sub_path, ceph_save_dir = get_ceph_path(output_dir, ceph_args)
            local_save_dir = os.path.join('/dev/shm', sub_path)
            Path(local_save_dir).mkdir(parents=True, exist_ok=True)
        else:
            local_save_dir = output_dir
        tag_name = "checkpoint-%s" % model_name
        model.save_checkpoint(save_dir=local_save_dir, tag=tag_name, client_state=client_state)

        if ceph_args['use_ceph_checkpoint'] and ceph_args['local_rank'] == 0:
            try:
                assert has_client == True, "petrel_client is not installed!!!"
                ckpt_shm_dir = os.path.join(local_save_dir, tag_name)
                ckpt_petrel_dir = os.path.join(ceph_save_dir, tag_name)
                for f_name in os.listdir(ckpt_shm_dir):
                    f_shm_path = os.path.join(ckpt_shm_dir, f_name)
                    f_petrel_path = os.path.join(ckpt_petrel_dir, f_name)
                    with open(f_shm_path, 'rb') as f:
                        print(f"Upload checkpoint at {f_petrel_path}", flush=True)
                        client.put(f_petrel_path, f)
                        print("Finish! Will remove the original files!", flush=True)
                    os.remove(f_shm_path)
            except Exception as e:
                print(f'Fail to upload or delete {f_shm_path} with error {e}')


def auto_load_model(args, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, ceph_args={'use_ceph_checkpoint': False}):
    output_dir = Path(args.output_dir)

    if ceph_args['use_ceph_checkpoint']:
        assert has_client == True, "petrel_client is not installed!!!"
        sub_path, ceph_save_dir = get_ceph_path(output_dir, ceph_args)
        if loss_scaler is not None:
            # torch.amp
            if args.test_best and args.eval:
                args.resume = os.path.join(ceph_save_dir, 'checkpoint-best.pth')
            elif check_ceph_exists(os.path.join(ceph_save_dir, 'checkpoint-latest.pth')):
                args.resume = os.path.join(ceph_save_dir, 'checkpoint-latest.pth')
            elif args.auto_resume and len(args.resume) == 0:
                all_checkpoints = fnmatch.filter(list(client.list(ceph_save_dir)), 'checkpoint-*')
                all_checkpoints = [
                    os.path.join(ceph_save_dir, ckpt_path)
                    for ckpt_path in all_checkpoints
                ]
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('-')[-1].split('.')[0]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)

            if args.resume:
                with io.BytesIO(client.get(args.resume)) as buffer:
                    checkpoint = torch.load(buffer, map_location='cpu')
                model_without_ddp.load_state_dict(checkpoint['model'])
                print("Resume checkpoint %s" % args.resume)
                if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    args.start_epoch = checkpoint['epoch'] + 1
                    if hasattr(args, 'model_ema') and args.model_ema:
                        _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                    if 'scaler' in checkpoint:
                        loss_scaler.load_state_dict(checkpoint['scaler'])
                    print("With optim & sched!")
        else:
            # deepspeed, only support '--auto_resume'.
            flag = False
            if args.test_best and args.eval:
                try:
                    load_specific_ceph_model(
                        model, model_ema, args, sub_path, ceph_save_dir, 
                        model_name='best', ceph_args=ceph_args
                    )
                    flag = True
                except Exception:
                    print('No best model')
            if not flag:
                try:
                    load_specific_ceph_model(
                        model, model_ema, args, sub_path, ceph_save_dir, 
                        model_name='latest', ceph_args=ceph_args
                    )
                    flag = True
                except Exception:
                    print('No latest model')
            if not flag:
                try:
                    load_specific_ceph_model(
                        model, model_ema, args, sub_path, ceph_save_dir, 
                        model_name='best', ceph_args=ceph_args
                    )
                    flag = True
                except Exception:
                    print('No best model')
            if not flag: 
                all_checkpoints = fnmatch.filter(list(client.list(ceph_save_dir)), 'checkpoint-*')
                all_checkpoints = [
                    os.path.join(ceph_save_dir, ckpt_path)
                    for ckpt_path in all_checkpoints
                ]
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('-')[-1].split('.')[0]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    load_specific_ceph_model(
                        model, model_ema, args, sub_path, ceph_save_dir, 
                        model_name=latest_ckpt, ceph_args=ceph_args
                    )
                else:
                    print('No other models')
    else:
        if loss_scaler is not None:
            # torch.amp
            if args.test_best and args.eval:
                args.resume = os.path.join(output_dir, 'checkpoint-best.pth')
            elif os.path.exists(os.path.join(output_dir, 'checkpoint-latest.pth')):
                args.resume = os.path.join(output_dir, 'checkpoint-latest.pth')
            elif args.auto_resume and len(args.resume) == 0:
                import glob
                all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*.pth'))
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('-')[-1].split('.')[0]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    args.resume = os.path.join(output_dir, 'checkpoint-%d.pth' % latest_ckpt)
            print("Auto resume checkpoint: %s" % args.resume)

            if args.resume:
                checkpoint = torch.load(args.resume, map_location='cpu')
                model_without_ddp.load_state_dict(checkpoint['model'])
                print("Resume checkpoint %s" % args.resume)
                if 'optimizer' in checkpoint and 'epoch' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    args.start_epoch = checkpoint['epoch'] + 1
                    if hasattr(args, 'model_ema') and args.model_ema:
                        _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                    if 'scaler' in checkpoint:
                        loss_scaler.load_state_dict(checkpoint['scaler'])
                    print("With optim & sched!")
        else:
            # deepspeed, only support '--auto_resume'.
            flag = False
            if args.test_best and args.eval:
                try:
                    load_specific_model(model, model_ema, args, output_dir, model_name='best')
                    flag = True
                except Exception:
                    print('No best model')
            if not flag:
                try:
                    load_specific_model(model, model_ema, args, output_dir, model_name='latest')
                    flag = True
                except Exception:
                    print('No latest model')
            if not flag:
                try:
                    load_specific_model(model, model_ema, args, output_dir, model_name='best')
                    flag = True
                except Exception:
                    print('No best model')
            if not flag: 
                import glob
                all_checkpoints = glob.glob(os.path.join(output_dir, 'checkpoint-*'))
                latest_ckpt = -1
                for ckpt in all_checkpoints:
                    t = ckpt.split('-')[-1].split('.')[0]
                    if t.isdigit():
                        latest_ckpt = max(int(t), latest_ckpt)
                if latest_ckpt >= 0:
                    load_specific_model(model, model_ema, args, output_dir, model_name=latest_ckpt)
                else:
                    print('No other models')


def load_specific_model(model, model_ema, args, output_dir, model_name):
    args.resume = os.path.join(output_dir, f'checkpoint-{model_name}')
    print(f"Auto resume the {model_name} checkpoint")
    _, client_states = model.load_checkpoint(args.output_dir, tag=f'checkpoint-{model_name}')
    args.start_epoch = client_states['epoch'] + 1
    if model_ema is not None:
        if args.model_ema:
            _load_checkpoint_for_ema(model_ema, client_states['model_ema'])


def check_ceph_exists(ceph_path):
    return list(client.list(ceph_path)) > 0


def load_specific_ceph_model(model, model_ema, args, sub_path, ceph_save_dir, model_name, ceph_args):
    tag_name = f'checkpoint-{model_name}'
    args.resume = os.path.join(ceph_save_dir, tag_name)
    print(f"Auto resume checkpoint: {args.resume}", flush=True)
    shm_resume_dir = os.path.join('/dev/shm', sub_path, tag_name)
    Path(shm_resume_dir).mkdir(parents=True, exist_ok=True)
    
    if ceph_args['local_rank'] == 0:
        for f_name in client.list(args.resume):
            ckpt_petrel_path = os.path.join(args.resume, f_name)
            ckpt_shm_path = os.path.join(shm_resume_dir, f_name)
            print(f"Download model from {ckpt_petrel_path}", flush=True)
            with open(ckpt_shm_path, 'wb') as f:
                f.write(memoryview(client.get(ckpt_petrel_path)))
            print("Finish downloading!", flush=True)

    torch.distributed.barrier()
    
    _, client_states = model.load_checkpoint(os.path.join('/dev/shm', sub_path), tag=f'checkpoint-{model_name}')
    args.start_epoch = client_states['epoch'] + 1
    if model_ema is not None:
        if args.model_ema:
            _load_checkpoint_for_ema(model_ema, client_states['model_ema'])

    if ceph_args['local_rank'] == 0:
        try:
            for root, dirs, files in os.walk(shm_resume_dir):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
                os.rmdir(root)
        except Exception as e:
            print(f'Fail to clean {shm_resume_dir} with error {e}')


def create_ds_config(args):
    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        0.9,
                        0.999
                    ],
                    "eps": 1e-8
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 7,
                "loss_scale_window": 128
            }
        }

        writer.write(json.dumps(ds_config, indent=2))


def create_internvideo2_lp_ds_config(args):
    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": 1000,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        args.opt_betas[0],
                        args.opt_betas[1]
                    ],
                    "eps": args.opt_eps
                }
            },
            "fp16": {
                "enabled": not args.bf16,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 500,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "bf16": {
                "enabled": args.bf16
            },
        }
        if args.clip_grad is not None:
            ds_config.update({'gradient_clipping': args.clip_grad})

        writer.write(json.dumps(ds_config, indent=2))


# stolen from https://github.com/baaivision/EVA/blob/7389aeeec97c056fc8424fa6b78f35c6f1b07d0d/EVA-02/asuka/utils.py#L529C5-L599C54
def create_internvideo2_ds_config(args):
    args.deepspeed_config = os.path.join(args.output_dir, "deepspeed_config.json")
    with open(args.deepspeed_config, mode="w") as writer:
        ds_config = {
            "train_batch_size": args.batch_size * args.update_freq * get_world_size(),
            "train_micro_batch_size_per_gpu": args.batch_size,
            "steps_per_print": args.steps_per_print,
            "optimizer": {
                "type": "Adam",
                "adam_w_mode": True,
                "params": {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "bias_correction": True,
                    "betas": [
                        args.opt_betas[0],
                        args.opt_betas[1]
                    ],
                    "eps": args.opt_eps
                }
            },
            "fp16": {
                "enabled": not args.bf16,
                "loss_scale": 0,
                "initial_scale_power": 16,
                "loss_scale_window": 500,
                "hysteresis": 2,
                "min_loss_scale": 1
            },
            "bf16": {
                "enabled": args.bf16
            },
            "amp": {
                "enabled": False,
                "opt_level": "O2"
            },
            "flops_profiler": {
                "enabled": True,
                "profile_step": -1,
                "module_depth": -1,
                "top_modules": 1,
                "detailed": True,
            },
            "zero_allow_untested_optimizer": True
        }

        if args.clip_grad is not None:
            ds_config.update({'gradient_clipping': args.clip_grad})

        if args.zero_stage == 1:
            ds_config.update(
                {
                    "zero_optimization": {
                        "stage": 1, 
                        "reduce_bucket_size": 5e8,
                    }
                }
            )
        elif args.zero_stage == 2:
            ds_config.update(
                {
                    "zero_optimization": {
                        "stage": 2, 
                        "contiguous_gradients": True,
                        "overlap_comm": True,
                        "reduce_scatter": True,
                        "reduce_bucket_size": 5e8,
                        "allgather_bucket_size": 5e8,
                        "cpu_offload": False,
                    }
                }
            )
        elif args.zero_stage == 3:
            ds_config.update(
                {
                    "zero_optimization": {
                        "stage": 3,
                        "contiguous_gradients": True,
                        "overlap_comm": True,
                        "reduce_scatter": True,
                        "reduce_bucket_size": 5e4,
                        "allgather_bucket_size": 5e4,
                        "cpu_offload": False,
                        "stage3_max_live_parameters": 1e5,
                        "stage3_max_reuse_distance": 1e5,
                    },
                }
            )
        elif args.zero_stage > 3:
            raise NotImplementedError()

        opt_lower = args.opt.lower()
        if opt_lower != 'adamw': del ds_config['optimizer']

        writer.write(json.dumps(ds_config, indent=2))


def multiple_samples_collate(batch, fold=False):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels, video_idx, extra_data = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    video_idx = [item for sublist in video_idx for item in sublist]
    inputs, labels, video_idx, extra_data = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(extra_data),
    )
    if fold:
        return [inputs], labels, video_idx, extra_data
    else:
        return inputs, labels, video_idx, extra_data


def multiple_pretrain_samples_collate(batch, fold=False):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    process_data, mask = zip(*batch)
    process_data = [item for sublist in process_data for item in sublist]
    mask = [item for sublist in mask for item in sublist]
    process_data, mask = (
        default_collate(process_data),
        default_collate(mask),
    )
    if fold:
        return [process_data], mask
    else:
        return process_data, mask