import torch


# def forzen_param(model):
#     for name, param in model.named_parameters():
#         if 'mlm_score' in name or 'vtm_score' in name or 'mpp_score' in name:
#             param.requires_grad = True
#         else:
#             param.requires_grad = False
#     return True


def forzen_param(model):
    flag = False
    for name, param in model.named_parameters():
        if '10' in name:
            flag = True
        param.requires_grad = flag
    return True