import os
import PIL
import tempfile
import compressai
import torch
import torch.nn.functional as F
import torchaudio
import opuspy
import encodec
import datasets
from io import BytesIO
from torchvision import transforms
try:
    import tfci
except:
    import urllib.request 
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/tensorflow/compression/master/models/tfci.py",
        "tfci.py")
    import tfci

def jpeg_compress(img):
    w = img.width
    h = img.height
    with BytesIO() as f:
        img.save(f, format='JPEG',quality=5)
        img = f.getvalue()
    bpp = 8*len(img)/(w*h)
    img = PIL.Image.open(BytesIO(img))
    return img,bpp

def webp_compress(img):
    w = img.width
    h = img.height
    with BytesIO() as f:
        img.save(f, format='WEBP',quality=0)
        img = f.getvalue()
    bpp = 8*len(img)/(w*h)
    img = PIL.Image.open(BytesIO(img))
    return img,bpp

def pad(x, p=2**6):
    h, w = x.size(2), x.size(3)
    pad, _ = compressai.ops.compute_padding(h, w, min_div=p)
    return F.pad(x, pad, mode="constant", value=0)

def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    _, unpad = compressai.ops.compute_padding(h, w, out_h=H, out_w=W)
    return F.pad(x, unpad, mode="constant", value=0)

def nn_compress(img,net,device):
    w = img.width
    h = img.height
    
    if (img.mode == 'L') | (img.mode == 'CMYK') | (img.mode == 'RGBA'):
        rgbimg = PIL.Image.new("RGB", img.size)
        rgbimg.paste(img)
        img = rgbimg
    
    x = transforms.ToTensor()(img).unsqueeze(0)
    x = pad(x)
    x = x.to(device)
    
    with torch.no_grad():
        compressed = net.compress(x)
        recovered  = net.decompress(compressed['strings'],shape=compressed['shape'])
    recovered['x_hat'].clamp_(0, 1);
    img = transforms.ToPILImage()(recovered['x_hat'].squeeze())
    bpp = 8*sum(len(str[0]) for str in compressed['strings'])/(w*h)
    return img,bpp

def hific_lo_compress(input_img):
    with tempfile.NamedTemporaryFile('wb', delete=True) as f_input:
        with tempfile.NamedTemporaryFile('wb', delete=True) as f_compressed:
            with tempfile.NamedTemporaryFile('wb', delete=True) as f_output:
                input_img.save(f_input.name, format='png')
                tfci.compress("hific-lo",f_input.name,f_compressed.name)
                size = os.path.getsize(f_compressed.name)
                bpp = size / (input_img.width*input_img.height)
                tfci.decompress(f_compressed.name,f_output.name)
                output_img = PIL.Image.open(f_output.name)
                rgbimg = PIL.Image.new("RGB", output_img.size)
                rgbimg.paste(output_img)
                output_img = rgbimg
    return output_img,bpp

def mp3_compress(audio,fs):
    N = audio.numel()
    L = audio.shape[1]
    with BytesIO() as f:
        torchaudio.save(f, audio, sample_rate=fs, format="mp3", compression=8.0)
        f.seek(0)
        bps = 8*len(f.getvalue())/N
        audio = torchaudio.load(f,format="mp3")[0]
        audio = audio[:,1106:]
        audio = audio[:,:L]
    return audio,bps

def opus_compress(audio,fs):
    N = audio.numel()
    audio = audio.clamp(min=-1.0,max=1.0)
    audio = 2**15*audio
    audio = audio.to(torch.int16)
    audio = audio.permute((1,0))
    audio = audio.numpy()
    with tempfile.NamedTemporaryFile('w+b', delete=True) as f:
        opuspy.write(
            path=f.name,
            waveform_tc=audio,
            sample_rate=48000,
            bitrate=6000,
            signal_type=0,
            encoder_complexity=10
        )
        audio = torchaudio.load(f,format="opus")[0]
        bps = 8*os.path.getsize(f.name)/N
    return audio, bps

def encodec_compress(audio,fs,model,device):
    N = audio.numel()
    N_samples = audio.shape[-1]
    audio = encodec.utils.convert_audio(audio,fs,model.sample_rate,model.channels)
    with torch.no_grad():
        x = audio.unsqueeze(0).to(device)
        encoded_frames = model.encode(x)
        audio = model.decode(encoded_frames)
    codes = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)
    audio = audio.detach().cpu()
    audio = encodec.utils.convert_audio(audio,model.sample_rate,fs,model.channels)
    audio = audio[0,:,0:N_samples]
    bps = codes.numel()*model.bits_per_codebook/N
    return audio,bps

def hf_audio_encode(audio,fs):
    with tempfile.NamedTemporaryFile() as f:
        torchaudio.save(f.name, audio, sample_rate=fs, format="wav")
        bytes = f.file.read(-1)
        encoded = datasets.Audio(
            sampling_rate=fs,
            mono=False,
            decode=False
        ).encode_example(value={"path" : None, "bytes": bytes})
    return encoded