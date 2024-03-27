VisionEncoders = dict()
VisionEncoders["beit"] = dict(
    name="beit_base",
    pretrained="microsoft/beit-base-patch16-224-pt22k-ft22k",
    d_model=768,
)
VisionEncoders["beit_large"] = dict(
    name="beit_large",
    pretrained="microsoft/beit-large-patch16-224-pt22k-ft22k",
    d_model=1024,
)

TextEncoders = dict()
TextEncoders["bert"] = dict(
    name="bert_base",
    pretrained="bert-base-uncased",
    config="configs/config_bert.json",
    d_model=768,
    fusion_layer=9,
)
TextEncoders["bert_large"] = dict(
    name="bert_large",
    pretrained="bert-large-uncased",
    config="configs/config_bert_large.json",
    d_model=1024,
    fusion_layer=19,
)
