# pretrain dataset
## video
from CoTrain.datamodules.video.webvid_datamodule import WEBVIDDataModule
from CoTrain.datamodules.video.webvid10m_datamodule import WEBVID10MDataModule
from CoTrain.datamodules.video.howto100m_datamodule import HT100MDataModule
from CoTrain.datamodules.video.youtube_datamodule import YOUTUBEDataModule
from CoTrain.datamodules.video.yttemporal_datamodule import YTTemporalMDataModule
## image
from CoTrain.datamodules.image.cc3m_datamodule import CC3MDataModule
from CoTrain.datamodules.image.cc12m_datamodule import CC12MDataModule
from CoTrain.datamodules.image.yfcc15m_datamodule import YFCC15MDataModule
from CoTrain.datamodules.image.laion400m_datamodule import LAION400MDataModule
from CoTrain.datamodules.image.vg_caption_datamodule import VisualGenomeCaptionDataModule
from CoTrain.datamodules.image.coco_caption_karpathy_datamodule import CocoCaptionKarpathyDataModule
from CoTrain.datamodules.image.conceptual_caption_datamodule import ConceptualCaptionDataModule
from CoTrain.datamodules.image.sbu_datamodule import SBUCaptionDataModule
from CoTrain.datamodules.image.mix100m_datamodule import MIX100MDataModule
# finetune dataset
## image
from CoTrain.datamodules.image.f30k_caption_karpathy_datamodule import F30KCaptionKarpathyDataModule
from CoTrain.datamodules.image.vqav2_datamodule import VQAv2DataModule
from CoTrain.datamodules.image.nlvr2_datamodule import NLVR2DataModule
from CoTrain.datamodules.image.vcr_datamodule import VCRDataModule
## video
from CoTrain.datamodules.video.msrvtt_datamodule import MSRVTTDataModule
from CoTrain.datamodules.video.msrvttqa_datamodule import MSRVTTQADataModule
from CoTrain.datamodules.video.msrvtt_choice_datamodule import MSRVTTChoiceDataModule
from CoTrain.datamodules.video.msvd_datamodule import MSVDDataModule
from CoTrain.datamodules.video.msvdqa_datamodule import MSVDQADataModule
from CoTrain.datamodules.video.ego4d_datamodule import Ego4DDataModule
from CoTrain.datamodules.video.tvqa_datamodule import TVQADataModule
from CoTrain.datamodules.video.lsmdc_choice_datamodule import LSMDCChoiceDataModule
from CoTrain.datamodules.video.ego4d_choice_datamodule import EGO4DChoiceDataModule
from CoTrain.datamodules.video.tgif_datamodule import TGIFDataModule
from CoTrain.datamodules.video.tgifqa_datamodule import TGIFQADataModule
from CoTrain.datamodules.video.didemo_datamodule import DIDEMODataModule
from CoTrain.datamodules.video.hmdb51_datamodule import HMDB51DataModule
from CoTrain.datamodules.video.ucf101_datamodule import UCF101DataModule
from CoTrain.datamodules.video.k400_datamodule import K400DataModule
from CoTrain.datamodules.video.lsmdc_datamodule import LSMDCDataModule
from CoTrain.datamodules.video.k400_video_datamodule import K400VideoDataModule
from CoTrain.datamodules.image.activitynet_datamodule import ActivityNetDataModule

_datamodules = {
    # image
    "vg": VisualGenomeCaptionDataModule,
    "f30k": F30KCaptionKarpathyDataModule,
    "coco": CocoCaptionKarpathyDataModule,
    "gcc": ConceptualCaptionDataModule,
    "sbu": SBUCaptionDataModule,
    "vqa": VQAv2DataModule,
    "nlvr2": NLVR2DataModule,
    "cc3m": CC3MDataModule,
    "cc12m": CC12MDataModule,
    'yfcc15m': YFCC15MDataModule,
    'laion400m': LAION400MDataModule,
    'vcr': VCRDataModule,
    'mix100m': MIX100MDataModule,
    # video
    'howto100m': HT100MDataModule,
    'youtube': YOUTUBEDataModule,
    'webvid': WEBVIDDataModule,
    'webvid10m': WEBVID10MDataModule,
    'msrvtt': MSRVTTDataModule,
    'msrvttqa': MSRVTTQADataModule,
    'msrvtt_choice': MSRVTTChoiceDataModule,
    'msvd': MSVDDataModule,
    'msvdqa': MSVDQADataModule,
    'ego4d': Ego4DDataModule,
    'tvqa': TVQADataModule,
    'lsmdc_choice': LSMDCChoiceDataModule,
    'ego4d_choice': EGO4DChoiceDataModule,
    'yttemporal': YTTemporalMDataModule,
    'tgif': TGIFDataModule,
    "tgifqa": TGIFQADataModule,
    'didemo': DIDEMODataModule,
    'hmdb51': HMDB51DataModule,
    'ucf101': UCF101DataModule,
    'k400': K400DataModule,
    'lsmdc': LSMDCDataModule,
    'activitynet': ActivityNetDataModule,
    'k400_video': K400VideoDataModule,
}
