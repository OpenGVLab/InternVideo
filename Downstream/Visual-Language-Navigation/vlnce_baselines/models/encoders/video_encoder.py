from vlnce_baselines.models.videomae import volume_transforms, video_transforms, modeling_finetune, utils
from vlnce_baselines.models.videomae.get_args import get_args
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

from habitat import logger

from collections import OrderedDict

class VideoRGBEcnoder(nn.Module):
    r"""
    Takes in observations and produces an embedding of the rgb component.

    Args:
        observation_space: The observation_space of the agent
        output_size: The size of the embedding vector
        device: torch.device
    """

    def __init__(
        self,
        observation_space,
        output_size,
        model_name,
        device,
        spatial_output: bool = False,
    ):
        super().__init__()
        self.device = device

        args, ds_init = get_args()
        
        if "Large" in model_name:
            args.model = 'vit_large_patch16_224'
            args.finetune = 'pretrained/VideoMAE/vit_l_hybrid_pt_800e.pth'
        elif "Base" in model_name:
            args.model = 'vit_base_patch16_224'
            args.finetune = 'pretrained/VideoMAE/vit_b_hybrid_pt_800e.pth'

        model = create_model(
            args.model,
            pretrained=False,
            num_classes=args.nb_classes,
            all_frames=args.num_frames * args.num_segments,
            tubelet_size=args.tubelet_size,
            drop_rate=args.drop,
            drop_path_rate=args.drop_path,
            attn_drop_rate=args.attn_drop_rate,
            drop_block_rate=None,
            use_mean_pooling=args.use_mean_pooling,
            init_scale=args.init_scale,
        )

        
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load ckpt from %s" % args.finetune)
        checkpoint_model = None
        for model_key in args.model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict

        # interpolate position embedding
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
            num_patches = model.patch_embed.num_patches # 
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

            # height (== width) for the checkpoint position embedding 
            orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(args.num_frames // model.patch_embed.tubelet_size)) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int((num_patches // (args.num_frames // model.patch_embed.tubelet_size) )** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(-1, args.num_frames // model.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, args.num_frames // model.patch_embed.tubelet_size, new_size, new_size, embedding_size) 
                pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

        utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
        
        self.model = model

        # disable gradients for resnet, params frozen
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.resnet_layer_size = 768
        self.spatial_output = spatial_output

        if not self.spatial_output:
            self.output_shape = (output_size,)
            # self.fc = nn.Linear(768, output_size)
            # self.activation = nn.ReLU()
        else:

            class SpatialAvgPool(nn.Module):
                def forward(self, x):
                    x = F.adaptive_avg_pool2d(x, (4, 4))

                    return x

            self.model.avgpool = SpatialAvgPool()
            self.model.fc = nn.Sequential()

            self.spatial_embeddings = nn.Embedding(4 * 4, 64)

            self.output_shape = (
                self.resnet_layer_size + self.spatial_embeddings.embedding_dim,
                4,
                4,
            )

        self.data_transform = video_transforms.Compose([
                video_transforms.Resize(args.short_side_size, interpolation='bilinear'),
                video_transforms.CenterCrop(size=(args.input_size, args.input_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            ])

    @property
    def is_blind(self):
        return self._n_input_rgb == 0

    def forward(self, observations):
        r"""Sends RGB observation through the TorchVision ResNet50 pre-trained
        on ImageNet. Sends through fully connected layer, activates, and
        returns final embedding.
        """

        # permute tensor to dimension [BATCH x CHANNEL x HEIGHT x WIDTH]
        # rgb_observations = observations["rgb"].permute(0, 3, 1, 2)
        video_rgb_batch = torch.vstack([self.data_transform(obs[k])[None,...] for obs in observations["video_rgbs"] for k in obs.keys()]).cuda()

        if self.spatial_output:
            features = self.model.get_spatial_features(video_rgb_batch)
            features = self.model.avgpool(features)
            b, c, h, w = features.size()

            spatial_features = (
                self.spatial_embeddings(
                    torch.arange(
                        0,
                        self.spatial_embeddings.num_embeddings,
                        device=features.device,
                        dtype=torch.long,
                    )
                )
                .view(1, -1, h, w)
                .expand(b, self.spatial_embeddings.embedding_dim, h, w)
            )

            return torch.cat([features, spatial_features], dim=1)#.to(self.device)
        else:
            return self.model.forward_features(video_rgb_batch)