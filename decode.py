import os
import time
import struct
import warnings
import argparse
import subprocess
import tracemalloc
import fpzip
import torch
import rasterio
import numpy as np
from encode import interpolate
import logger
import Codecs


warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_CODECS = ['JPEG2000', 'JPEGXL', 'HEVC', 'VVC'] #


def read_image_header(bitstream):
    ptr = 0
    n_bytes_header = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1
    BASE_CODEC = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1
    s = struct.unpack('<f',bitstream[ptr: ptr + 4])[0]
    ptr += 4
    D = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
    ptr += 1
    # Q = struct.unpack('<f',bitstream[ptr: ptr + 4])[0]
    # ptr += 4
    H = int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
    ptr += 2
    W = int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
    ptr += 2
    lsr_bytes = int.from_bytes(bitstream[ptr: ptr + 3], byteorder='big', signed=False)

    return n_bytes_header, BASE_CODEC, s, D, H, W, lsr_bytes
    
        
def test(bin_path, org_path=None):
    # tracemalloc.start()

    dirname, basename = os.path.split(bin_path)
    filename = os.path.splitext(basename)[0]
    logger.create_logger(dirname, 'decode.txt')
    logger.log.info(f'Binstream: {bin_path}')
    start_time = time.time()
    with open(bin_path, 'rb') as fin: bitstream = fin.read()

    n_bytes_header, BASE_CODEC, s, D, H, W, lsr_bytes = read_image_header(bitstream)
    bitstream = bitstream[n_bytes_header:]
    
    recon_path = f'{dirname}/{basename[:-4]}_recon.tif'

    sub_lsr_bitstream = bitstream[:lsr_bytes]
    sub_lsr_bitstream_path = f'{dirname}/{filename}_lsr.bin'
    with open(sub_lsr_bitstream_path, 'wb') as f_out: f_out.write(sub_lsr_bitstream)
    bitstream = bitstream[lsr_bytes:]
    
    base_recon_path = f'{dirname}/{filename}_base_recon.tif'
    base_bin_path = f'{dirname}/{filename}_base.bin'
    with open(base_bin_path, 'wb') as f_out: f_out.write(bitstream)
    BASE_CODEC = BASE_CODECS[BASE_CODEC]
    Codecs.decode(base_bin_path, base_recon_path, method=BASE_CODEC)

    dataset = rasterio.open(base_recon_path)
    base = dataset.read()  # CHW or HW
    subprocess.call(f'rm -f {base_recon_path}', shell=True)
    base = base.reshape((-1, base.shape[-2], base.shape[-1])) # CHW
    C = base.shape[0]
    H1 = np.linspace(0, H - 1, int(H // s), dtype=int)
    W1 = np.linspace(0, W - 1, int(W // s), dtype=int)
    if s > 1:
        base_data = interpolate(base, H1, W1, H, W, C)
    else:
        base_data = base.astype(np.float32)
    
    feature_dim = C * (2 * D + 1) ** 2
    features = np.zeros((H, W, feature_dim), dtype=np.float32) # 
    if D > 0:
        base_data_pad = np.pad(base_data, 
                               ((0, 0), (D, D), (D, D)),
                               mode='reflect'
                               ).transpose(1, 2, 0) # (H+2D)(W+2D)C
        colors = np.lib.stride_tricks.sliding_window_view(base_data_pad, (2 * D + 1, 2 * D + 1), 
                                                          axis=(0, 1))
        features = colors.reshape((H, W, -1))
    else:
        features = base_data.transpose(1, 2, 0)
    X = features.reshape(H * W, feature_dim)

    with open(sub_lsr_bitstream_path,'rb') as f: compressed_bytes = f.read()
    beta = fpzip.decompress(compressed_bytes, order='C')[0][0][0].reshape(-1, C)

    image = (X @ beta).reshape(H, W, C) 
    image = np.transpose(image, axes=(2, 0, 1))   
    image = np.clip(image, a_min=0, a_max=10000) #
    image = image.round().astype(np.uint16)
    Codecs.write_tiff_with_rasterio(recon_path, image)

    logger.log.info(f'Recon: {recon_path}')
    subprocess.call(f'rm -f {base_bin_path}', shell=True)
    subprocess.call(f'rm -f {base_bin_path}.aux.xml', shell=True)
    subprocess.call(f'rm -f {sub_lsr_bitstream_path}', shell=True)


    end_time = time.time()
    logger.log.info(f'Time elapsed: {end_time - start_time}')

    # current, peak = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    # logger.log.info(f"Current memory usage: {current / 10**6:.2f} MB")
    # logger.log.info(f"Peak memory usage: {peak / 10**6:.2f} MB")

    if org_path is not None:
        org_img = rasterio.open(org_path).read() #
        rec_img = rasterio.open(recon_path).read()
        bytes = os.path.getsize(bin_path)
        mse_value = np.mean((org_img.astype(np.float32) - rec_img.astype(np.float32)) ** 2) #
        logger.log.info(f"MSE: {mse_value}")
        peak = np.max(org_img.astype(np.float32)) # 4095 for 12-bits ARAD images
        psnr = 10 * np.log10(peak ** 2 / mse_value)
        logger.log.info(f"PSNR: {psnr}")   
        n_subpixels = np.prod(org_img.shape)
        logger.log.info(f"Total size: {bytes} bytes, bpsp={bytes * 8 / n_subpixels}")
        if True: # False: # Delete the reconstructed image?
            subprocess.call(f'rm -f {recon_path}', shell=True)
    
    return bitstream


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LoCaCF-HSIC')
    parser.add_argument('-i', '--bin_path', type=str, help='binstream path')
    parser.add_argument('-org', '--org_path', type=str, default=None, help='org path')
    args = parser.parse_args()

    test(args.bin_path, args.org_path)
