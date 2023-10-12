import os
import PIL
import tempfile
import compressai
import torch
import torch.nn.functional as F
import torchaudio
import opuspy
import encodec
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

def jpeg_compress(sample):
    img = sample['image']
    w = img.width
    h = img.height
    with BytesIO() as f:
        img.save(f, format='JPEG',quality=5)
        img = f.getvalue()
    sample['image'] = PIL.Image.open(BytesIO(img))
    sample['bpp'] = 8*len(img)/(w*h)
    return sample

def webp_compress(sample):
    img = sample['image']
    w = img.width
    h = img.height
    with BytesIO() as f:
        img.save(f, format='WEBP',quality=0)
        img = f.getvalue()
    sample['image'] = PIL.Image.open(BytesIO(img))
    sample['bpp'] = 8*len(img)/(w*h)
    return sample

def pad(x, p=2**6):
    h, w = x.size(2), x.size(3)
    pad, _ = compressai.ops.compute_padding(h, w, min_div=p)
    return F.pad(x, pad, mode="constant", value=0)

def crop(x, size):
    H, W = x.size(2), x.size(3)
    h, w = size
    _, unpad = compressai.ops.compute_padding(h, w, out_h=H, out_w=W)
    return F.pad(x, unpad, mode="constant", value=0)

def nn_compress(sample,net,device):
    img = sample['image']
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
    sample['image'] = transforms.ToPILImage()(recovered['x_hat'].squeeze())
    sample['bpp'] = 8*sum(len(str[0]) for str in compressed['strings'])/(w*h)
    return sample

def hific_lo_compress(sample):
    input_img = sample['image']
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
                
    sample['image'] = output_img
    sample['bpp'] = bpp
    return sample

def mp3_compress(sample):
    audio = sample['audio']['array'].unsqueeze(0)
    fs = sample['audio']['sampling_rate']
    with BytesIO() as f:
        torchaudio.save(f, audio, sample_rate=fs, format="mp3", compression=8.0)
        f.seek(0)
        sample['bps'] = len(f.getvalue())/len(audio[0])
        audio = torchaudio.load(f,format="mp3")
    sample['audio']['array'] = audio[0][0]
    return sample

def opus_compress(sample):
    audio = sample['audio']['array'].unsqueeze(1)
    audio = (2**15*audio.clamp(min=-1.0,max=1.0)).to(torch.int16).numpy()
    fs = sample['audio']['sampling_rate']
    with tempfile.NamedTemporaryFile('wb', delete=True) as f:
        opuspy.write(
            path=f.name,
            waveform_tc=audio,
            sample_rate=fs,
            bitrate=6000,
            signal_type=2,
            encoder_complexity=10
        )
        audio,fs = opuspy.read(f.name)
        audio = torch.tensor(audio).to(torch.float32)/(2**15)
        audio = audio.transpose(0,1)
        sample['bps'] = os.path.getsize(f.name)/len(audio[0])
    sample['audio']['array'] = audio[0]
    return sample


def encodec_compress(sample,model,device):
    audio = sample['audio']['array'].unsqueeze(0)
    fs = sample['audio']['sampling_rate']
    audio = encodec.utils.convert_audio(audio,fs,model.sample_rate,model.channels)
    with torch.no_grad():
        x = audio.unsqueeze(0).to(device)
        encoded_frames = model.encode(x)
        sample['audio']['array'] = model.decode(encoded_frames).mean(dim=[0,1])
        sample['bps'] = (40/8)*sum(fi[0].shape[2] for fi in encoded_frames)/sample['audio']['array'].shape[0]
    return sample