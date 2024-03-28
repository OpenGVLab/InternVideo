try:
    from petrel_client.client import Client

    client = Client()

    # Disable boto logger
    import logging
    logging.getLogger('boto3').setLevel(logging.WARNING)
    logging.getLogger('botocore').setLevel(logging.WARNING)
    logging.getLogger('nose').setLevel(logging.WARNING)
except:
    client = None

# == pretrain data
# = image
from .image.vg_caption_dataset import VisualGenomeCaptionDataset
from .image.coco_caption_karpathy_dataset import CocoCaptionKarpathyDataset
from .image.sbu_caption_dataset import SBUCaptionDataset
from .image.cc3m import CC3MDataset
from .image.cc12m import CC12MDataset
from .image.yfcc15m import YFCC15MDataset
from .image.laion400m import LAION400MDataset
from .image.conceptual_caption_dataset import ConceptualCaptionDataset
from .image.mix100m import MIX100MDataset
# = video
from .video.webvid import WEBVIDDataset
from .video.webvid10m import WEBVID10MDataset
from .video.howto100m import HT100MDataset
from .video.youtube import YOUTUBEDataset
from .video.yttemporal import YTTemporalDataset
# == downstream data
# = image
from .image.f30k_caption_karpathy_dataset import F30KCaptionKarpathyDataset
from .image.vqav2_dataset import VQAv2Dataset
from .image.nlvr2_dataset import NLVR2Dataset
from .image.vcr import VCRDataset
# = video
from .video.msrvtt import MSRVTTDataset
from .video.msrvttqa import MSRVTTQADataset
from .video.msrvtt_choice import MSRVTTChoiceDataset
from .video.msvd import MSVDDataset
from .video.lsmdc_dataset import LSMDCDataset
from .video.msvdqa import MSVDQADataset
from .video.ego4d import Ego4DDataset
from .video.tvqa import TVQADataset
from .video.lsmdc_choice import LSMDCChoiceDataset
from .video.ego4d_choice import EGO4DChoiceDataset
from .video.tgif import TGIFDataset
from .video.tgifqa import TGIFQADataset
from .video.didemo import DIDEMODataset
from .video.hmdb51 import HMDB51Dataset
from .video.ucf101 import UCF101Dataset
from .video.k400 import K400Dataset
from .video.activitynet import ActivityNetDataset
from .video.k400_video import K400VideoDataset