# Recurrent VLN-BERT, 2020, by Yicong.Hong@anu.edu.au

from pytorch_transformers import (BertConfig, BertTokenizer)

def get_vlnbert_models(config=None):
    config_class = BertConfig

    from vlnce_baselines.models.vlnbert.vlnbert_PREVALENT import VLNBert
    model_class = VLNBert
    # model_name_or_path = 'data/mydata/snap/VLNBERT-PREVALENT-final/state_dict/best_val_unseen'
    model_name_or_path = 'pretrained/Prevalent/pretrained_model/pytorch_model.bin'
    vis_config = config_class.from_pretrained('pretrained/Prevalent/bert-base-uncased')
    vis_config.img_feature_dim = 2176
    vis_config.img_feature_type = ""
    vis_config.vl_layers = 4
    vis_config.la_layers = 9
    visual_model = model_class.from_pretrained(model_name_or_path, config=vis_config)

    return visual_model
