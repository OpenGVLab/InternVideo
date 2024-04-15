import os as __os  # add "__" if not want to be exported
from copy import deepcopy as __deepcopy


# ============== pretraining datasets=================
available_corpus = dict(
    # pretraining image datasets
    cc3m=dict(
        anno_path="your_path", 
        data_root="",
        media_type="image"
    ),
    cc12m=dict(
        anno_path="your_path", 
        data_root="",
        media_type="image"
    ),
    sbu=dict(
        anno_path="your_path", 
        data_root="",
        media_type="image"
    ),
    vg=dict(
        anno_path="your_path", 
        data_root="",
        media_type="image",
        jump_filter=True
    ),
    coco=dict(
        anno_path="your_path", 
        data_root="",
        media_type="image",
        jump_filter=True
    ),
    laion_2b=dict(
        anno_path="your_path",
        data_root="",
        media_type="image",
        jump_filter=True
    ),
    laion_coco=dict(
        anno_path="your_path",
        data_root="",
        media_type="image",
        jump_filter=True
    ),
    laion_pop=dict(
        anno_path="your_path",
        data_root="",
        media_type="image",
        jump_filter=True
    ),
    # pretraining video datasets
    webvid_fuse_10m=dict(
        anno_path="your_path", 
        data_root="",
        media_type="video",
        jump_filter=True
    ),
    internvid_v1=dict(
        anno_path="your_path",
        data_root="",
        media_type="video",
        jump_filter=True
    ),
    internvid_v2_avs_private=dict( 
        anno_path="your_path",
        data_root="",
        media_type="audio_video",
        read_clip_from_video=False,
        read_audio_from_video=True,
        zero_audio_padding_for_video=True,
        caption_augmentation=dict(caption_sample_type='avs_all'),
        jump_filter=True
    ),
    webvid=dict(
        anno_path="your_path",
        data_root="",
        media_type="video"
    ),
    webvid_10m=dict(
        anno_path="your_path",
        data_root="",
        media_type="video",
    ),
    # audio-text
    wavcaps_400k=dict(
        anno_path="your_path",
        data_root="",
        media_type="audio"
    ),
    # debug
    cc3m_debug=dict(
        anno_path="your_path",
        data_root="",
        media_type="image"
    ),
    webvid_debug=dict(
        anno_path="your_path",
        data_root="",
        media_type="video"
    )
)

available_corpus["pretrain_example_data_1B"] = [
    available_corpus['cc3m'], 
    available_corpus['webvid']
]

available_corpus["pretrain_example_data_6B"] = [
    available_corpus['cc3m'], 
    available_corpus['webvid'], 
    available_corpus['internvid_v2_avs_private']
]

available_corpus["data_25m"] = [
    available_corpus["webvid_10m"],
    available_corpus["cc3m"],
    available_corpus["coco"],
    available_corpus["vg"],
    available_corpus["sbu"],
    available_corpus["cc12m"],
]

available_corpus["debug"] = [
    available_corpus["cc3m_debug"],
    available_corpus["webvid_debug"],
]


# ============== for validation =================
available_corpus["msrvtt_1k_test"] = dict(
    anno_path="your_path",
    data_root="",
    media_type="video"
)

available_corpus["didemo_ret_test"] = dict(
    anno_path="your_path",
    data_root="",
    media_type="video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    max_txt_l=64
)

available_corpus["anet_ret_val"] = dict(
    anno_path="your_path",
    data_root="",
    media_type="video",
    is_paragraph_retrieval=True,
    max_txt_l = 150
)

available_corpus["lsmdc_ret_test_1000"] = dict(
    anno_path="your_path",
    data_root="",
    media_type="video"
)

available_corpus["vatex_ch_ret_val"] = dict(
    anno_path="your_path",
    data_root="",
    media_type="video"
)

available_corpus["vatex_en_ret_val"] = dict(
    anno_path="your_path",
    data_root="",
    media_type="video"
)

available_corpus["k400_act_val"] = dict(
    anno_path="your_path",
    data_root="",
    is_act_rec=True,
)

available_corpus["k600_act_val"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    is_act_rec=True,
)

available_corpus["k700_act_val"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    is_act_rec=True,
)

available_corpus["mit_act_val"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    is_act_rec=True,
)

available_corpus["ucf101_act_val"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    is_act_rec=True,
)

available_corpus["hmdb51_act_val"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    is_act_rec=True,
)

available_corpus["ssv2_mc_val"] = dict(
    anno_path="your_path",
    data_root="",
    media_type="video",
)

available_corpus["charades_mc_test"] = dict(
    anno_path="your_path",
    data_root="",
    media_type="video",
)


available_corpus["anet_ret_train"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    is_paragraph_retrieval=True,
    max_txt_l = 150
)

available_corpus["didemo_ret_train"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    max_txt_l=64 
)

available_corpus["didemo_ret_val"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    is_paragraph_retrieval=True,
    trimmed30=True,
    max_txt_l=64
)

available_corpus["lsmdc_ret_train"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    max_txt_l=96
)

available_corpus["lsmdc_ret_val"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    max_txt_l=96
)

available_corpus["msrvtt_ret_train9k"] = dict(
    anno_path="your_path",
    data_root="",
    media_type="video",
)

available_corpus["msrvtt_ret_test1k"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
)

available_corpus["msvd_ret_train"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    max_txt_l=64,
    has_multi_txt_gt=True
)

available_corpus["msvd_ret_val"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    max_txt_l=64
)

available_corpus["msvd_ret_test"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    max_txt_l=64
)


available_corpus["vatex_en_ret_train"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="video",
    has_multi_txt_gt=True
)


# audio-text

available_corpus["audiocaps_ret_train"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="audio",
)

available_corpus["audiocaps_ret_test"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="audio",
)


available_corpus["clothov1_ret_train"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="audio",
)

available_corpus["clothov1_ret_test"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="audio",
)

available_corpus["clothov2_ret_train"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="audio",
)

available_corpus["clothov2_ret_test"] = dict(
    anno_path="your_path", 
    data_root="",
    media_type="audio",
)