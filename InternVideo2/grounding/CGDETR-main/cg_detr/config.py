import os
import time
import torch
import argparse

from utils.basic_utils import mkdirp, load_json, save_json, make_zipfile, dict_to_markdown
import shutil

class BaseOptions(object):
    saved_option_filename = "opt.json"
    ckpt_filename = "model.ckpt"
    tensorboard_log_dir = "tensorboard_log"
    train_log_filename = "train.log.txt"
    eval_log_filename = "eval.log.txt"

    def __init__(self):
        self.parser = None
        self.initialized = False
        self.opt = None

    def initialize(self):
        self.initialized = True
        parser = argparse.ArgumentParser()
        parser.add_argument("--dset_name", type=str, choices=["hl", 'charadesSTA', ])
        parser.add_argument("--dset_domain", type=str, 
                            help="Domain to train for tvsum dataset. (Only used for tvsum and youtube-hl)")
        
        parser.add_argument("--eval_split_name", type=str, default="val",
                            help="should match keys in video_duration_idx_path, must set for VCMR")
        parser.add_argument("--debug", action="store_true",
                            help="debug (fast) mode, break all loops, do not load all data into memory.")
        parser.add_argument("--data_ratio", type=float, default=1.0,
                            help="how many training and eval data to use. 1.0: use all, 0.1: use 10%."
                                 "Use small portion for debug purposes. Note this is different from --debug, "
                                 "which works by breaking the loops, typically they are not used together.")
        parser.add_argument("--results_root", type=str, default="results")
        parser.add_argument("--exp_id", type=str, default=None, help="id of this run, required at training")
        parser.add_argument("--seed", type=int, default=2018, help="random seed")
        parser.add_argument("--device", type=int, default=0, help="0 cuda, -1 cpu")
        parser.add_argument("--num_workers", type=int, default=0,
                            help="num subprocesses used to load the data, 0: use main process")
        parser.add_argument("--no_pin_memory", action="store_true",
                            help="Don't use pin_memory=True for dataloader. "
                                 "ref: https://discuss.pytorch.org/t/should-we-set-non-blocking-to-true/38234/4")

        # training config
        parser.add_argument("--lr", type=float, default=2e-4, help="learning rate")
        parser.add_argument("--lr_drop", type=int, default=800, help="drop learning rate to 1/10 every lr_drop epochs")
        parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
        parser.add_argument("--n_epoch", type=int, default=200, help="number of epochs to run")
        parser.add_argument("--max_es_cnt", type=int, default=200,
                            help="number of epochs to early stop, use -1 to disable early stop")
        parser.add_argument("--bsz", type=int, default=32, help="mini-batch size")
        parser.add_argument("--eval_bsz", type=int, default=100,
                            help="mini-batch size at inference, for query")
        parser.add_argument("--eval_epoch", type=int, default=5,
                            help="inference epoch")
        parser.add_argument("--grad_clip", type=float, default=0.1, help="perform gradient clip, -1: disable")
        parser.add_argument("--eval_untrained", action="store_true", help="Evaluate on un-trained model")
        parser.add_argument("--resume", type=str, default=None,
                            help="checkpoint path to resume or evaluate, without --resume_all this only load weights")
        parser.add_argument("--resume_all", action="store_true",
                            help="if --resume_all, load optimizer/scheduler/epoch as well")
        parser.add_argument("--start_epoch", type=int, default=None,
                            help="if None, will be set automatically when using --resume_all")

        # Data config
        parser.add_argument("--max_q_l", type=int, default=-1)
        parser.add_argument("--max_v_l", type=int, default=-1)
        parser.add_argument("--clip_length", type=float, default=2)
        parser.add_argument("--max_windows", type=int, default=5)

        parser.add_argument("--train_path", type=str, default=None)
        parser.add_argument("--eval_path", type=str, default=None,
                            help="Evaluating during training, for Dev set. If None, will only do training, ")
        parser.add_argument("--no_norm_vfeat", action="store_true", help="Do not do normalize video feat")
        parser.add_argument("--no_norm_tfeat", action="store_true", help="Do not do normalize text feat")
        parser.add_argument("--v_feat_dirs", type=str, nargs="+",
                            help="video feature dirs. If more than one, will concat their features. "
                                 "Note that sub ctx features are also accepted here.")
        parser.add_argument("--t_feat_dir", type=str, help="text/query feature dir")
        parser.add_argument("--a_feat_dir", type=str, help="audio feature dir")
        parser.add_argument("--v_feat_dim", type=int, help="video feature dim")
        parser.add_argument("--t_feat_dim", type=int, help="text/query feature dim")
        parser.add_argument("--a_feat_dim", type=int, help="audio feature dim")
        parser.add_argument("--ctx_mode", type=str, default="video_tef")

        # Model config
        parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                            help="Type of positional embedding to use on top of the image features")
        # * Transformer
        parser.add_argument('--enc_layers', default=3, type=int,
                            help="Number of encoding layers in the transformer")
        parser.add_argument('--dec_layers', default=3, type=int,
                            help="Number of decoding layers in the transformer")
        parser.add_argument('--t2v_layers', default=2, type=int,
                            help="Number of decoding layers in the transformer")
        parser.add_argument('--sent_layers', default=1, type=int,
                            help="Number of decoding layers in the transformer")
        parser.add_argument('--moment_layers', default=1, type=int,
                            help="Number of decoding layers in the transformer")
        parser.add_argument('--dummy_layers', default=2, type=int,
                            help="Number of encoding layers in the transformer")
        parser.add_argument('--dim_feedforward', default=1024, type=int,
                            help="Intermediate size of the feedforward layers in the transformer blocks")
        parser.add_argument('--hidden_dim', default=256, type=int,
                            help="Size of the embeddings (dimension of the transformer)")
        parser.add_argument('--input_dropout', default=0.5, type=float,
                            help="Dropout applied in input")
        parser.add_argument('--dropout', default=0.1, type=float,
                            help="Dropout applied in the transformer")
        parser.add_argument("--txt_drop_ratio", default=0, type=float,
                            help="drop txt_drop_ratio tokens from text input. 0.1=10%")
        parser.add_argument("--use_txt_pos", action="store_true", help="use position_embedding for text as well.")
        parser.add_argument('--nheads', default=8, type=int,
                            help="Number of attention heads inside the transformer's attentions")
        parser.add_argument('--num_queries', default=10, type=int,
                            help="Number of query slots")
        parser.add_argument('--num_dummies', default=45, type=int,
                            help="Number of dummy tokens")
        parser.add_argument('--total_prompts', default=10, type=int,
                            help="Number of query slots")
        parser.add_argument('--num_prompts', default=1, type=int,
                            help="Number of dummy tokens")
        parser.add_argument('--pre_norm', action='store_true')
        # other model configs
        parser.add_argument("--n_input_proj", type=int, default=2, help="#layers to encoder input")
        parser.add_argument("--contrastive_hdim", type=int, default=64, help="dim for contrastive embeddings")
        parser.add_argument("--temperature", type=float, default=0.07, help="temperature nce contrastive_align_loss")
        # Loss

        parser.add_argument("--saliency_margin", type=float, default=0.2)
        parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                            help="Disables auxiliary decoding losses (loss at each layer)")
        parser.add_argument("--span_loss_type", default="l1", type=str, choices=['l1', 'ce'],
                            help="l1: (center-x, width) regression. ce: (st_idx, ed_idx) classification.")
        parser.add_argument("--contrastive_align_loss", action="store_true",
                            help="Disable contrastive_align_loss between matched query spans and the text.")
        # * Matcher
        parser.add_argument('--set_cost_span', default=10, type=float,
                            help="L1 span coefficient in the matching cost")
        parser.add_argument('--set_cost_giou', default=1, type=float,
                            help="giou span coefficient in the matching cost")
        parser.add_argument('--set_cost_class', default=4, type=float,
                            help="Class coefficient in the matching cost")

        # * Loss coefficients
        parser.add_argument("--lw_saliency", type=float, default=1.,
                            help="weight for saliency loss, set to 0 will ignore")
        parser.add_argument("--lw_wattn", type=float, default=1.,
                            help="weight for saliency loss, set to 0 will ignore")
        parser.add_argument("--lw_ms_align", type=float, default=1.,
                            help="weight for saliency loss, set to 0 will ignore")
        parser.add_argument("--lw_distill", type=float, default=1.,
                            help="weight for saliency loss, set to 0 will ignore")
        parser.add_argument('--span_loss_coef', default=10, type=float)
        parser.add_argument('--giou_loss_coef', default=1, type=float)
        parser.add_argument('--label_loss_coef', default=4, type=float)
        parser.add_argument('--eos_coef', default=0.1, type=float,
                            help="Relative classification weight of the no-object class")
        parser.add_argument("--contrastive_align_loss_coef", default=0.0, type=float)

        parser.add_argument("--no_sort_results", action="store_true",
                            help="do not sort results, use this for moment query visualization")
        parser.add_argument("--max_before_nms", type=int, default=10)
        parser.add_argument("--max_after_nms", type=int, default=10)
        parser.add_argument("--conf_thd", type=float, default=0.0, help="only keep windows with conf >= conf_thd")
        parser.add_argument("--nms_thd", type=float, default=-1,
                            help="additionally use non-maximum suppression "
                                 "(or non-minimum suppression for distance)"
                                 "to post-processing the predictions. "
                                 "-1: do not use nms. [0, 1]")
        self.parser = parser

    def display_save(self, opt):
        args = vars(opt)
        # Display settings
        print(dict_to_markdown(vars(opt), max_str_len=120))
        # Save settings
        if not isinstance(self, TestOptions):
            option_file_path = os.path.join(opt.results_dir, self.saved_option_filename)  # not yaml file indeed
            save_json(args, option_file_path, save_pretty=True)

    def parse(self, a_feat_dir=None):
        if not self.initialized:
            self.initialize()
        opt = self.parser.parse_args()

        if opt.debug:
            opt.results_root = os.path.sep.join(opt.results_root.split(os.path.sep)[:-1] + ["debug_results", ])
            opt.num_workers = 0

        if isinstance(self, TestOptions):
            # modify model_dir to absolute path
            # opt.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", opt.model_dir)
            opt.model_dir = os.path.dirname(opt.resume)
            if a_feat_dir is not None:
                opt.a_feat_dir = a_feat_dir
            saved_options = load_json(os.path.join(opt.model_dir, self.saved_option_filename))
            for arg in saved_options:  # use saved options to overwrite all BaseOptions args.
                if arg not in ["results_root", "num_workers", "nms_thd", "debug",  # "max_before_nms", "max_after_nms"
                               "max_pred_l", "min_pred_l",
                               "resume", "resume_all", "no_sort_results"]:
                    setattr(opt, arg, saved_options[arg])
            # opt.no_core_driver = True
            if opt.eval_results_dir is not None:
                opt.results_dir = opt.eval_results_dir
        else:
            if opt.exp_id is None:
                raise ValueError("--exp_id is required for at a training option!")

            ctx_str = opt.ctx_mode + "_sub" if any(["sub_ctx" in p for p in opt.v_feat_dirs]) else opt.ctx_mode
            opt.results_dir = os.path.join(opt.results_root,
                                           "-".join([opt.dset_name, ctx_str, opt.exp_id,
                                                     str(opt.enc_layers) + str(opt.dec_layers) + str(opt.t2v_layers) + str(opt.moment_layers) + str(opt.dummy_layers) + str(opt.sent_layers),
                                                     'ndum_' + str(opt.num_dummies), 'nprom_' + str(opt.num_prompts) + '_' + str(opt.total_prompts)]))
            mkdirp(opt.results_dir)
            save_fns = ['cg_detr/model.py', 'cg_detr/transformer.py']
            for save_fn in save_fns:
                shutil.copyfile(save_fn, os.path.join(opt.results_dir, os.path.basename(save_fn)))

            # save a copy of current code
            code_dir = os.path.dirname(os.path.realpath(__file__))
            code_zip_filename = os.path.join(opt.results_dir, "code.zip")
            make_zipfile(code_dir, code_zip_filename,
                         enclosing_dir="code",
                         exclude_dirs_substring="results",
                         exclude_dirs=["results", "debug_results", "__pycache__"],
                         exclude_extensions=[".pyc", ".ipynb", ".swap"], )

        self.display_save(opt)

        opt.ckpt_filepath = os.path.join(opt.results_dir, self.ckpt_filename)
        opt.train_log_filepath = os.path.join(opt.results_dir, self.train_log_filename)
        opt.eval_log_filepath = os.path.join(opt.results_dir, self.eval_log_filename)
        opt.tensorboard_log_dir = os.path.join(opt.results_dir, self.tensorboard_log_dir)
        opt.device = torch.device("cuda" if opt.device >= 0 else "cpu")
        opt.pin_memory = not opt.no_pin_memory

        opt.use_tef = "tef" in opt.ctx_mode
        opt.use_video = "video" in opt.ctx_mode
        if not opt.use_video:
            opt.v_feat_dim = 0
        if opt.use_tef:
            opt.v_feat_dim += 2

        self.opt = opt
        return opt


class TestOptions(BaseOptions):
    """add additional options for evaluating"""

    def initialize(self):
        BaseOptions.initialize(self)
        # also need to specify --eval_split_name
        self.parser.add_argument("--eval_id", type=str, help="evaluation id")
        self.parser.add_argument("--eval_results_dir", type=str, default=None,
                                 help="dir to save results, if not set, fall back to training results_dir")
        self.parser.add_argument("--model_dir", type=str,
                                 help="dir contains the model file, will be converted to absolute path afterwards")

