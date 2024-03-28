# Code for "ActionCLIP: ActionCLIP: A New Paradigm for Action Recognition"
# arXiv:
# Mengmeng Wang, Jiazheng Xing, Yong Liu

import torch
import CoTrain.modules.InternVideo as internvideo
from CoTrain.datasets import K400VideoDataset


def text_prompt(data = K400VideoDataset.classes(), prompt_type='all'):
    if prompt_type == 'all':
        text_aug = [f"a photo of action {{}}", f"a picture of action {{}}", f"Human action of {{}}", f"{{}}, an action",
                    f"{{}} this is an action", f"{{}}, a video of action", f"Playing action of {{}}", f"{{}}",
                    f"Playing a kind of action, {{}}", f"Doing a kind of action, {{}}", f"Look, the human is {{}}",
                    f"Can you recognize the action of {{}}?", f"Video classification of {{}}", f"A video of {{}}",
                    f"The man is {{}}", f"The woman is {{}}"]
    elif prompt_type == 'single':
        text_aug = [f"A video of {{}}"]
    elif prompt_type == 'single_doing':
        text_aug = [f"A person is doing {{}}"]
    elif prompt_type == 'no':
        text_aug = [f"{{}}"]
    print('-' * 80)
    print('Prompt:')
    print(text_aug)
    print('-' * 80)
    text_dict = {}
    num_text_aug = len(text_aug)

    for ii, txt in enumerate(text_aug):
        text_dict[ii] = torch.cat([internvideo.tokenize(txt.format(c), truncate=True) for c in data])

    classes = torch.cat([v for _, v in text_dict.items()])

    return classes, num_text_aug, text_dict