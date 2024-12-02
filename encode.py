import os
import time
import struct
import warnings
import argparse
import subprocess
import tracemalloc
import fpzip
import rasterio
import numpy as np
# https://docs.scipy.org/doc/scipy/reference/interpolate.html
from scipy.interpolate import interpn, RectBivariateSpline 
import logger
import Codecs


warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
BASE_CODECS = ['JPEG2000', 'JPEGXL', 'HEVC', 'VVC'] #


def interpolate(downsampled, H1, W1, H, W, C):
    grid_y, grid_x = np.linspace(0, H - 1, H), np.linspace(0, W - 1, W)
    interpolated = np.zeros((C, H, W)) 
    for c in range(C):
        spline = RectBivariateSpline(H1, W1, downsampled[c])
        interpolated[c] = spline(grid_y, grid_x)
    
    # z, y, x = np.meshgrid(np.arange(C), np.arange(H), np.arange(W), indexing='ij')
    # interpn_method = 'linear' if C < 4 else 'cubic'
    # interpolated = interpn((np.arange(C), H1, W1), downsampled, 
    #                        (z.ravel(), y.ravel(), x.ravel()), 
    #                        method=interpn_method, bounds_error=False, fill_value=0)
    # interpolated = interpolated.reshape(C, H, W)

    return interpolated.astype(np.float32) 


def process(path, s, Q, D, output_path, BASE_CODEC):
    im = rasterio.open(path).read() #
    C, H, W = im.shape
    assert s > 1 or (s == 1 and Q < 100) # no lossless

    H1 = np.linspace(0, H - 1, int(H // s), dtype=int)
    W1 = np.linspace(0, W - 1, int(W // s), dtype=int)
    downsampled_data = im[:, H1[:, None], W1]

    base_bin_path = output_path.replace(".tif", ".bin")
    recon_path = output_path.replace(".tif", "_recon.tif")
    if BASE_CODEC in ['HEVC', 'VVC']: #
        output_path = output_path.replace(".tif", ".yuv")
        downsampled_data.tofile(output_path) # preprocessing to yuv400
    else:
        Codecs.write_tiff_with_rasterio(output_path, downsampled_data)
    NBITS = int(np.ceil(np.log2(downsampled_data.max() + 1)))
    Codecs.encode(output_path, base_bin_path, Q, BASE_CODEC,
                  NBITS, C, int(H // s), int(W // s))
    subprocess.call(f'rm -f {output_path}', shell=True)
    subprocess.call(f'rm -f {base_bin_path}.aux.xml', shell=True)
    if Q == 100:
        base_data_d = downsampled_data
    else:
        Codecs.decode(base_bin_path, recon_path, method=BASE_CODEC)
        dataset = rasterio.open(recon_path)
        base_data_d = dataset.read()
        subprocess.call(f'rm -f {recon_path}', shell=True)
    if s > 1:
        base_data = interpolate(base_data_d, H1, W1, H, W, C)
    else:
        base_data = base_data_d.astype(np.float32) 
    
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
    y = im.astype(np.float32).transpose(1, 2, 0).reshape(H * W, C)
    beta, _, _, _ = np.linalg.lstsq(X, y)
    
    return beta, H, W, C


def write_image_header(header_path, BASE_CODEC, s, D, H, W, lsr_bytes):
    n_bytes_header  = 0
    n_bytes_header += 1      # Number of bytes header
    n_bytes_header += 1      # BASE_CODECS.index
    n_bytes_header += 4      # s
    n_bytes_header += 1      # D
    # n_bytes_header += 4      # Q
    n_bytes_header += 2      # H
    n_bytes_header += 2      # W
    n_bytes_header += 3      # Number of bytes lsr
    byte_to_write   = b''
    byte_to_write  += n_bytes_header.to_bytes(1, byteorder='big', signed=False)
    byte_to_write  += BASE_CODECS.index(BASE_CODEC).to_bytes(1, byteorder='big', signed=False)
    byte_to_write  += struct.pack('<f', s)
    byte_to_write  += D.to_bytes(1, byteorder='big', signed=False)
    # byte_to_write  += struct.pack('<f', Q)
    byte_to_write  += H.to_bytes(2, byteorder='big', signed=False)
    byte_to_write  += W.to_bytes(2, byteorder='big', signed=False)
    byte_to_write += lsr_bytes.to_bytes(3, byteorder='big', signed=False)
    with open(header_path, 'wb') as fout: fout.write(byte_to_write)
    if n_bytes_header != os.path.getsize(header_path):
        raise ValueError(f'Invalid number of bytes in header! '
                         f'expected {n_bytes_header}, got {os.path.getsize(header_path)}')


def train(path, base_codec, s, Q, D, precision, output_dir='outputs'):
    # tracemalloc.start()
    filename = os.path.splitext(os.path.basename(path))[0]
    fs = '{}/{}_{}_s{}_Q{}_D{}_prec{}'
    output_dir = fs.format(output_dir, filename, base_codec, 
                           s, Q, D, precision)
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    logger.create_logger(output_dir, 'encode.txt')
    start_time = time.time()
    bitstream_path = f'{output_dir}/{filename}.bin'
    header_path = f'{bitstream_path}_header'
    output_path = f'{output_dir}/{filename}_base.tif'
    beta, H, W, C = process(path, s, Q, D, output_path, base_codec)
    n_subpixels = C * H * W

    compressed_bytes = fpzip.compress(beta.reshape(-1), precision=precision, order='C')

    lsr_bitstream_path = f'{output_dir}/{filename}_lsr.bin'
    with open(lsr_bitstream_path, 'wb') as f: f.write(compressed_bytes)
    lsr_bytes = os.path.getsize(lsr_bitstream_path)
    lsr_bpsp = lsr_bytes * 8 / n_subpixels
    logger.log.info(f'lsr: {lsr_bytes} bytes, bpsp={lsr_bpsp}')
    base_bitstream_path = f'{output_dir}/{filename}_base.bin'
    base_bytes = os.path.getsize(base_bitstream_path)
    base_bpsp = base_bytes * 8 / n_subpixels
    logger.log.info(f"base: {base_bytes} bytes: bpsp={base_bpsp}")

    subprocess.call(f'rm -f {header_path}', shell=True)
    write_image_header(header_path, base_codec, s, D, H, W, lsr_bytes)
    subprocess.call(f'rm -f {bitstream_path}', shell=True)
    subprocess.call(f'cat {header_path} >> {bitstream_path}', shell=True)
    subprocess.call(f'rm -f {header_path}', shell=True)
    subprocess.call(f'cat {lsr_bitstream_path} >> {bitstream_path}', shell=True)
    subprocess.call(f'rm -f {lsr_bitstream_path}', shell=True)
    subprocess.call(f'cat {base_bitstream_path} >> {bitstream_path}', shell=True)
    subprocess.call(f'rm -f {base_bitstream_path}', shell=True)

    end_time = time.time()
    logger.log.info(f'Time elapsed: {end_time - start_time}')

    # current, peak = tracemalloc.get_traced_memory()
    # tracemalloc.stop()
    # logger.log.info(f"Current memory usage: {current / 10**6:.2f} MB")
    # logger.log.info(f"Peak memory usage: {peak / 10**6:.2f} MB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LoCaCF-HSIC')
    parser.add_argument('-i', '--path', type=str,
                        help='path of input tif or img file')
    parser.add_argument('-o', '--output_dir', default='outputs', type=str,
                        help='output dir')
    parser.add_argument('-codec', '--base_codec', default='JPEGXL', type=str,
                        help='Base codec (Default: JPEGXL)')
    parser.add_argument('-s', '--s', type=float, default=1.0,
                        help=' (default: 1.0)')
    parser.add_argument('-Q', '--Q', type=float, default=36.0,
                        help=' (default: 100.0)')
    parser.add_argument('-D', '--D', type=int, default=0,
                        help='#neighbors (2D+1)^2, default D: 0')
    parser.add_argument('-prec', '--precision', type=int, default=32,
                        help=' (default: 32)')
    args = parser.parse_args()

    print(args)

    train(args.path, args.base_codec, args.s, args.Q, args.D, args.precision, args.output_dir)