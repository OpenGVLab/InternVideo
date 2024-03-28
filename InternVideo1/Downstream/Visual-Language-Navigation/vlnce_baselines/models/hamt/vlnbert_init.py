import torch


def get_tokenizer(args):
    from transformers import AutoTokenizer
    if args.dataset == 'rxr' or args.tokenizer == 'xlm':
        cfg_name = 'xlm-roberta-base'
    else:
        cfg_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer

def get_vlnbert_models(config=None):
    
    from transformers import PretrainedConfig
    from vlnce_baselines.models.hamt.vilmodel_cmt import NavCMT

    model_class = NavCMT

    model_name_or_path = config.pretrained_path
    new_ckpt_weights = {}
    if model_name_or_path is not None:
        ckpt_weights = torch.load(model_name_or_path, map_location='cpu')
        for k, v in ckpt_weights.items():
            if k.startswith('module'):
                new_ckpt_weights[k[7:]] = v
            else:
                # add next_action in weights
                if k.startswith('next_action'):
                    k = 'bert.' + k
                new_ckpt_weights[k] = v
    
    if config.task_type == 'r2r':
        cfg_name = 'pretrained/Prevalent/bert-base-uncased'
    elif config.task_type == 'rxr':
        cfg_name = 'pretrained/xlm-roberta-base'
    vis_config = PretrainedConfig.from_pretrained(cfg_name)

    if config.task_type == 'r2r':
        vis_config.image_feat_size = 768
        vis_config.max_action_steps = 50 
    elif config.task_type == 'rxr':
        vis_config.type_vocab_size = 2
        vis_config.image_feat_size = 512
        vis_config.max_action_steps = 100
    
    # vis_config.image_feat_size = 768
    vis_config.depth_feat_size = 128
    vis_config.angle_feat_size = 4
    vis_config.num_l_layers = 9
    vis_config.num_r_layers = 0
    vis_config.num_h_layers = 0
    vis_config.num_x_layers = 4
    vis_config.hist_enc_pano = True
    vis_config.num_h_pano_layers = 2

    vis_config.fix_lang_embedding = config.fix_lang_embedding
    vis_config.fix_hist_embedding = config.fix_hist_embedding
    vis_config.fix_obs_embedding = config.fix_obs_embedding

    vis_config.update_lang_bert = not vis_config.fix_lang_embedding
    vis_config.output_attentions = True
    vis_config.pred_head_dropout_prob = 0.1

    vis_config.no_lang_ca = False
    vis_config.act_pred_token = 'ob_txt'
    # vis_config.max_action_steps = 50 
    # vis_config.max_action_steps = 100
    
    visual_model = model_class.from_pretrained(
        pretrained_model_name_or_path=None, 
        config=vis_config, 
        state_dict=new_ckpt_weights)
        
    return visual_model
