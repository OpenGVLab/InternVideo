import os
import torch
import librosa
import torchaudio
from BEATs import BEATs, BEATsConfig

# load the pre-trained checkpoints
checkpoint = torch.load('/mnt/petrelfs/yujiashuo/model/beats/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')

cfg = BEATsConfig(checkpoint['cfg'])
BEATs_model = BEATs(cfg)
BEATs_model.load_state_dict(checkpoint['model'])
BEATs_model.eval()
BEATs_model = BEATs_model.cuda()

root_path = '/mnt/petrelfs/yujiashuo/dataset/WavCaps/Zip_files/FreeSound/FreeSound_flac/'
audio_ls = os.listdir(root_path)[:1000]
sr = 16000
# extract the the audio representation
for audio_name in audio_ls:
    audio_path = os.path.join(root_path, audio_name)
    audio_input_16khz, _ = librosa.load(audio_path, sr=sr)
    audio_input_16khz = audio_input_16khz[:160000] if len(audio_input_16khz) > 160000 else audio_input_16khz
    audio_input_16khz = torch.from_numpy(audio_input_16khz).unsqueeze(0)
    audio_input_16khz = audio_input_16khz.expand(2, audio_input_16khz.size(-1))
    print(audio_input_16khz.shape)
    padding_mask = torch.zeros(audio_input_16khz.shape).bool().cuda()
    fbank = BEATs_model.preprocess(audio_input_16khz).cuda()
    representation = BEATs_model(fbank, padding_mask=padding_mask)[0]
    # print(representation.max())