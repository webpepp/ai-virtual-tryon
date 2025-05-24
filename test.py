import torch
import torch.nn.functional as F
import kornia.geometry as tgm
import os
import cv2
import numpy as np

from networks import SegGenerator, GMM, ALIASGenerator
from utils import gen_noise, load_checkpoint

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint_dir = './checkpoints'
load_height, load_width = 1024, 768

class Opt:
    def __init__(self):
        self.load_height = 256
        self.load_width = 192
        self.ngf = 64
        self.init_type = 'normal'
        self.init_variance = 0.02
        self.num_upsampling_layers = 'normal'  # could be 'more' or 'most' as well
        self.norm_G = 'spectralinstance'
        self.semantic_nc = 13  # VITON-HD usually uses 13 semantic classes
        self.output_nc = 3     # RGB
        self.input_nc = 9      # standard input channels for try-on image + pose + mask        

opt = Opt()

# Instantiate models using the same opt instance
seg = SegGenerator(opt, input_nc=21, output_nc=13).to(device)
gmm = GMM(opt, inputA_nc=7, inputB_nc=3).to(device)
alias = ALIASGenerator(opt, input_nc=9).to(device)

# Load checkpoints
load_checkpoint(seg, os.path.join(checkpoint_dir, 'seg_final.pth'))
load_checkpoint(gmm, os.path.join(checkpoint_dir, 'gmm_final.pth'))
load_checkpoint(alias, os.path.join(checkpoint_dir, 'alias_final.pth'))

# Set models to eval mode
seg.eval()
gmm.eval()
alias.eval()

def preprocess_image(path, size=(load_width, load_height)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    img = img[:, :, ::-1].transpose(2, 0, 1) / 255.0
    img = torch.FloatTensor(img).unsqueeze(0).to(device)
    return img

def run_tryon(person_img_path, cloth_img_path):
    person = preprocess_image(person_img_path)
    cloth = preprocess_image(cloth_img_path)
    cloth_mask = (cloth.sum(dim=1, keepdim=True) > 0).float()

    gauss = tgm.image.GaussianBlur((15, 15), (3, 3)).to(device)

    # TODO: replace with real parsing & pose estimation
    parse_agnostic = torch.randn(1, 13, load_height, load_width).to(device)
    pose = torch.randn(1, 18, load_height, load_width).to(device)
    img_agnostic = person

    parse_input = torch.cat([
        F.interpolate(cloth * cloth_mask, (256, 192)),
        F.interpolate(cloth_mask, (256, 192)),
        F.interpolate(parse_agnostic, (256, 192)),
        F.interpolate(pose, (256, 192)),
        gen_noise((1, 1, 256, 192)).to(device)
    ], dim=1)

    seg_out = seg(parse_input)
    parse = gauss(F.interpolate(seg_out, size=(load_height, load_width))).argmax(dim=1)[:, None]
    parse_onehot = torch.zeros(1, 13, load_height, load_width).to(device).scatter_(1, parse, 1.0)

    gmm_input = torch.cat([
        F.interpolate(parse_onehot[:, 2:3], (256, 192)),
        F.interpolate(pose, (256, 192)),
        F.interpolate(img_agnostic, (256, 192))
    ], dim=1)

    _, grid = gmm(gmm_input, F.interpolate(cloth, (256, 192)))
    warped_c = F.grid_sample(cloth, grid, padding_mode='border')
    warped_cm = F.grid_sample(cloth_mask, grid, padding_mode='border')

    misalign = parse_onehot[:, 2:3] - warped_cm
    misalign[misalign < 0] = 0
    parse_div = torch.cat([parse_onehot, misalign], dim=1)
    parse_div[:, 2:3] -= misalign

    output = alias(torch.cat([img_agnostic, pose, warped_c], dim=1), parse_onehot, parse_div, misalign)
    result = (output[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

    os.makedirs("output", exist_ok=True)
    result_path = "output/result.png"
    cv2.imwrite(result_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

    return result_path
