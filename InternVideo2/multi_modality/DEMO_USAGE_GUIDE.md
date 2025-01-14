# DEMO_USAGE_GUIDE

## 1. Environment Configuration

We mainly follow [UMT](https://github.com/OpenGVLab/Unmasked_Teacher) to prepare the enviroment. Python, Pytorch, and CUDA can use updated versions. The following environment configuration has been tested and works correctly.

- **Python**: 3.8
- **Torch**: 2.4.1
- **CUDA**: 11.8

### Additional Dependencies Installation

The following additional dependencies need to be installed:

- [ninja](https://github.com/ninja-build/ninja)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)

Installation commands are as follows:

```bash
pip install ninja
pip install flash-attn --no-build-isolation
pip install deepspeed
```

>⚠️ **Note**: During the installation, the wheel needs to be built, which may take several hours and might fail. If the installation fails, it is recommended to try reinstalling.

## 2. Relative Import Error:

If you encounter issues related to relative imports while running the code, you can try changing them to absolute imports. The following tutorial is based on absolute imports.


### (a) Set the PYTHONPATH, using the  `multi_modality` folder as the base directory for absolute imports

```bash
export PYTHONPATH=.../InternVideo/InternVideo2/multi_modality:$PYTHONPATH
```

### (b) Modify file paths to accommodate PYTHONPATH

In `multi_modality/demo_config.py`, import files from the `multi_modality/utils` folder:

```python
from utils.easydict import EasyDict
```

In `multi_modality/demo/utils.py`, import files from the `multi_modality/models` folder:

```python
from models.backbones.internvideo2 import pretrain_internvideo2_1b_patch14_224
from models.backbones.bert.builder import build_bert
from models.criterions import get_sim
from models.backbones.internvideo2.pos_embed import interpolate_pos_embed_internvideo2_new
from models.backbones.bert.tokenization_bert import BertTokenizer
```

In `multi_modality/models/criterions.py`, import files from the `multi_modality/utils` folder:

```python
from utils.distributed import get_rank, get_world_size
from utils.easydict import EasyDict
```

## 3. Model weights that need to be downloaded and configured:
### InternVideo2-Stage2
#### (a) Bert-Large-Uncased (We only use its tokenizer)

This weight is automatically downloaded by default. If it cannot be downloaded properly, you will need to manually download the weights and configure the path.

Download link: [https://huggingface.co/google-bert/bert-large-uncased/tree/main](https://huggingface.co/google-bert/bert-large-uncased/tree/main)

In the `setup_internvideo2` function in `multi_modality/demo/utils.py`, configure the Bert weight. Replace `your_model_path` in the following with the path to the weights.

⚠️Note: `your_model_path` should not include the file name, just the folder path.

```Python
tokenizer = BertTokenizer.from_pretrained('your_model_path', local_files_only=True)
#tokenizer = BertTokenizer.from_pretrained(config.model.text_encoder.pretrained, local_files_only=Tru
```

#### (b) InternVideo2-Stage2_1B-224p-f4

Download link: [https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4)

In the `multi_modality/demo/internvideo2_stage2_config.py` file, configure the InternVideo2 weight. Replace `your_model_path` in the following with the path to the weights.

⚠️Note: `your_model_path` should include the file name.

```python
model = dict(
    model_cls="InternVideo2_Stage2",
    vision_encoder=dict(
        ...
        pretrained='your_model_path',
        ...
    )
)
```

### InternVideo2-CLIP

1. Download [chinese_alpaca_lora_7b](https://github.com/OpenGVLab/InternVL/tree/main/clip_benchmark/clip_benchmark/models/internvl_c_pytorch/chinese_alpaca_lora_7b) and set the `llama_path` and `tokenizer_path` in `config.py`.
2. Download [InternVideo2-stage2_1b-224p-f4.pt](https://huggingface.co/OpenGVLab/InternVideo2/blob/main/InternVideo2-stage2_1b-224p-f4.pt) and set `vision_ckpt_path` in `config.py`.
3. Download [internvl_c_13b_224px](https://huggingface.co/OpenGVLab/InternVL/blob/main/internvl_c_13b_224px.pth) and set `text_ckpt_path` in `config.py`.
4. Download [Our lora weight](https://huggingface.co/OpenGVLab/InternVideo2-CLIP-1B-224p-f8) and set `pretrained_path` in `config.py`.



### 4. Run
Output example of InternVideo2-Stage2-1B:

```plaintext
text: A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run. ~ prob: 0.7927
text: A playful dog and its owner wrestle in the snowy yard, chasing each other with joyous abandon. ~ prob: 0.1769
text: A pet dog excitedly runs through the snowy yard, chasing a toy thrown by its owner. ~ prob: 0.0291
text: A man in a gray hat and coat walks through the snowy yard, carefully navigating around the trees. ~ prob: 0.0006
text: A person dressed in a blue jacket shovels the snow-covered pavement outside their house. ~ prob: 0.0003
```



