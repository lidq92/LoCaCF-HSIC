import os
import csv
import warnings
import resource
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import h5py
import rasterio
import numpy as np
from rasterio.errors import NotGeoreferencedWarning


MAX_WORKERS = int( 2 * os.cpu_count() // 3) # 
soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
new_soft_limit = 65535 #
resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft_limit, hard_limit))
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)


# https://github.com/libjxl/libjxl/blob/main/lib/jxl/encode.cc#L1547
def jxl_encoder_distance_from_quality(quality):
    if quality >= 100.0:
        return 0.0
    elif quality >= 30.0:
        return 0.1 + (100 - quality) * 0.09 
    else:
        return 53.0 / 3000.0 * quality * quality - 23.0 / 20.0 * quality + 25.0


def write_tiff_with_rasterio(output_path: str, array: np.array):
    # The numpy array must be data with shape CHW
    count, height, width = array.shape[0], array.shape[1], array.shape[2]  # C, H, W
    transform = rasterio.transform.from_origin(0, 0, 1, 1)  # 
    with rasterio.open(
        output_path, 'w', driver='GTiff',
        height=height, width=width, count=count,
        dtype=array.dtype, transform=transform
    ) as dst:
        for i in range(count): dst.write(array[i, :, :], i + 1)


def cleanup_temp_files(files):
    for file in files: 
        if os.path.exists(file): os.remove(file)


def run_subprocess_commands(commands):
    for cmd in commands: subprocess.call(cmd, shell=True)


def encode(file_path, bin_path,  q=None, method='JPEGXL',
           NBITS=12, C=31, H=482, W=512):
    if method  == 'JPEG2000': # JP2OpenJPEG driver with gdal_translate 
        # (opj_compress does not support multispectral images, and one needs to set -I option to use the irreversible DWT 9-7)
        if q is None: q = 100
        encode_command = f'gdal_translate -q -of JP2OpenJPEG -co QUALITY={q} -co NBITS={NBITS} {file_path} {bin_path}'
        if q == 100: encode_command += ' -co REVERSIBLE=YES'
        run_subprocess_commands([
            encode_command,
            f'rm -f {bin_path}.aux.xml',
        ])

    if method == 'JPEGXL': # JPEGXL driver with gdal_translate
        if q is None: q = 100
        d = jxl_encoder_distance_from_quality(q)
        effort = 7 # 
        encode_command = f'gdal_translate -q -of JPEGXL -co EFFORT={effort} -co NBITS={NBITS} {file_path} {bin_path}'
        if d > 0: encode_command += f' -co DISTANCE={d}'
        run_subprocess_commands([
            encode_command,
            f'rm -f {bin_path}.aux.xml',
        ])
    
    if method == 'HEVC':
        """
        git clone https://vcgit.hhi.fraunhofer.de/jvet/HM.git
        mkdir build && cd build
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS_RELEASE="-O3 -march=native -DNDEBUG -Wno-error=array-bounds" \
            -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -DNDEBUG -Wno-error=array-bounds" -DHIGH_BITDEPTH=ON ..
        make -j

        Note: 
        --ReconFile= -> no reconstruction;
        --TemporalSubsampleRatio=1 avoids temporal subsampling; 
        --InputBitDepth={NBITS}
        --InternalBitDepth=0 default 0 results in the same value of InputBitDepth;
        --ConformanceWindowMode=1 same as VVC default;
        """
        if q is None or q == 100: q = 'lossl'
        intra = True
        HM_dir = './HM'
        cmd = f'{HM_dir}/bin/TAppEncoderStatic' \
            f' -c {HM_dir}/cfg/{"encoder_intra_main_rext.cfg" if intra else "encoder_randomaccess_main_rext.cfg"}' \
            f' --InputFile={file_path} ' \
            f' --BitstreamFile={bin_path}.hevc' \
            f' --ReconFile=' \
            f' --InputChromaFormat=400' \
            f' --TemporalSubsampleRatio=1' \
            f' --InputBitDepth={NBITS}' \
            f' --InternalBitDepth=0' \
            f' --ConformanceWindowMode=1' \
            f' --SourceWidth={W} --SourceHeight={H} --FramesToBeEncoded={C}' \
            f' --FrameRate=1 --Level=8.5 --Tier=high'  # 
        if q == 'lossl':
            cmd += f' -c {HM_dir}/cfg/lossless/lossless.cfg --QP=0' #
        else:
            # assert isinstance(q, int)
            q = int(q) #
            cmd += f' --QP={q}'

        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, check=True)
        byte_to_write   = b''
        n_bytes_header = 7
        byte_to_write += n_bytes_header.to_bytes(1, byteorder='big', signed=False)
        byte_to_write += C.to_bytes(2, byteorder='big', signed=False)
        byte_to_write += H.to_bytes(2, byteorder='big', signed=False)
        byte_to_write += W.to_bytes(2, byteorder='big', signed=False)
        header_path = bin_path.replace('.bin', '.header')
        with open(header_path, 'wb') as fout: fout.write(byte_to_write)    
        run_subprocess_commands([
            f'rm -f {bin_path}', 
            f'cat {header_path} >> {bin_path}',
            f'rm -f {header_path}',
            f'cat {bin_path}.hevc >> {bin_path}',
            f'rm -f {bin_path}.hevc',
        ])

    if method == 'VVC':
        """
        git clone https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM.git
        mkdir build && cd build
        cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS_RELEASE="-O3 -march=native -DNDEBUG" \
            -DCMAKE_CXX_FLAGS_RELEASE="-O3 -march=native -DNDEBUG" -DENABLE_HIGH_BITDEPTH=ON ..
        make -j
        """
        if q is None or q == 100: q = 'lossl'
        intra = True
        VVC_dir = './VVCSoftware_VTM'
        cmd = f'{VVC_dir}/bin/EncoderAppStatic' \
            f' -c {VVC_dir}/cfg/{"encoder_intra_vtm.cfg" if intra else "encoder_randomaccess_vtm.cfg"}' \
            f' --InputFile={file_path} ' \
            f' --BitstreamFile={bin_path}.vvc' \
            f' --ReconFile=' \
            f' --InputChromaFormat=400' \
            f' --TemporalSubsampleRatio=1' \
            f' --InputBitDepth={NBITS}' \
            f' --InternalBitDepth=0' \
            f' --SourceWidth={W} --SourceHeight={H} --FramesToBeEncoded={C}' \
            f' --FrameRate=1 --Level=15.5 --Tier=high'  # 
        if q == 'lossl':
            cmd += f' -c {VVC_dir}/cfg/lossless/lossless.cfg --QP=0'
        else:
            # assert isinstance(q, int)
            q = int(q) #
            cmd += f' --QP={q}'
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, check=True)
        byte_to_write   = b''
        n_bytes_header = 7
        byte_to_write += n_bytes_header.to_bytes(1, byteorder='big', signed=False)
        byte_to_write += C.to_bytes(2, byteorder='big', signed=False)
        byte_to_write += H.to_bytes(2, byteorder='big', signed=False)
        byte_to_write += W.to_bytes(2, byteorder='big', signed=False)
        header_path = bin_path.replace('.bin', '.header')
        with open(header_path, 'wb') as fout: fout.write(byte_to_write)    
        run_subprocess_commands([
            f'rm -f {bin_path}', 
            f'cat {header_path} >> {bin_path}',
            f'rm -f {header_path}',
            f'cat {bin_path}.vvc >> {bin_path}',
            f'rm -f {bin_path}.vvc',
        ])


def decode(bin_path, recon_path, method='JPEGXL'):
    if method in ['JPEG2000', 'JPEGXL']:
        subprocess.call(f"gdal_translate -q -of GTiff {bin_path} {recon_path}", shell=True)

    if method == 'HEVC':
        with open(bin_path, 'rb') as fin: bitstream = fin.read()
        ptr = 0
        n_bytes_header = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
        ptr += 1
        C =  int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
        ptr += 2
        H =  int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
        ptr += 2
        W =  int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
        ptr += 2
        bitstream = bitstream[ptr:]
        with open(f'{bin_path}.hevc', 'wb') as f_out: 
            f_out.write(bitstream)
        HM_dir = './HM'
        subprocess.run(
            f'{HM_dir}/bin/TAppDecoderStatic -b {bin_path}.hevc -o {recon_path}.yuv',
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, check=True)
        rec_mat = np.fromfile(f'{recon_path}.yuv', '<u2').reshape((C, H, W))
        subprocess.call(f'rm -f {bin_path}.hevc {recon_path}.yuv', shell=True)
        write_tiff_with_rasterio(recon_path, rec_mat)

    if method == 'VVC':
        with open(bin_path, 'rb') as fin: bitstream = fin.read()
        ptr = 0
        n_bytes_header = int.from_bytes(bitstream[ptr: ptr + 1], byteorder='big', signed=False)
        ptr += 1
        C =  int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
        ptr += 2
        H =  int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
        ptr += 2
        W =  int.from_bytes(bitstream[ptr: ptr + 2], byteorder='big', signed=False)
        ptr += 2
        bitstream = bitstream[ptr:]
        with open(f'{bin_path}.vvc', 'wb') as f_out: 
            f_out.write(bitstream)
        VVC_dir = './VVCSoftware_VTM'
        subprocess.run(
            f'{VVC_dir}/bin/DecoderAppStatic -b {bin_path}.vvc -o {recon_path}.yuv',
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, check=True)
        rec_mat = np.fromfile(f'{recon_path}.yuv', '<u2').reshape((C, H, W))
        subprocess.call(f'rm -f {bin_path}.vvc {recon_path}.yuv', shell=True)
        write_tiff_with_rasterio(recon_path, rec_mat)


def eval_RD(bin_path, recon_path, file_path):
    org_img = rasterio.open(file_path).read().astype(np.float32) # CHW
    recon_img = rasterio.open(recon_path).read().astype(np.float32) # CHW
    mse_value = np.mean((org_img - recon_img) ** 2) #
    peak = np.max(org_img) # 4095 for 12-bits ARAD images
    psnr = 10 * np.log10(peak ** 2 / mse_value)
    bits = 8 * os.path.getsize(bin_path)
    C, H, W = org_img.shape
    bpsp = bits / (C * H * W)

    return mse_value, psnr, bits, bpsp


def run_one_rate_point(method, file_path, q, i, NBITS, C, H, W):    
    directory = os.path.dirname(file_path)
    bin_path = f'{directory}/{method}_{i}_{q}.bin'
    recon_path = f'{directory}/{method}_{i}_{q}.tif'
    
    encode(file_path, bin_path, q, method, NBITS, C, H, W)
    decode(bin_path, recon_path, method)
    mse_value, psnr, bits, bpsp = eval_RD(bin_path, recon_path, file_path.replace('.yuv', '.tif'))
    
    subprocess.call(f'rm -f {bin_path} {recon_path}', shell=True)
    print(f"file: {file_path} @q={q}, MSE: {mse_value}, PSNR: {psnr}, bits: {bits}, bpsp: {bpsp}")
    
    return file_path.replace('.yuv', '.tif'), q, psnr, bpsp, mse_value, bits, i


def main(start_id=901, end_id_plus_1=951):
    file_paths = [
        f"data/ARAD/ARAD_1K_{idx:04d}.mat" for idx in range(start_id, end_id_plus_1)
    ]
    output_paths = [
        f"data/ARAD/ARAD_1K_{idx:04d}.tif" for idx in range(start_id, end_id_plus_1)
    ]
    output_paths2 = [
        f"data/ARAD/ARAD_1K_{idx:04d}.yuv" for idx in range(start_id, end_id_plus_1)
    ]

    NBITS = 12
    C, H, W = 31, 482, 512 #
    for i, file_path in enumerate(file_paths):
        if os.path.exists(output_paths[i]) and os.path.exists(output_paths2[i]):
            continue
        file = h5py.File(file_path, 'r')
        hsi = file['cube'][:].transpose(0, 2, 1)
        hsi = hsi.clip(min=0, max=1) # Handling ARAD_1K_0929
        org_img = np.round(hsi * (2 ** NBITS - 1)).astype('<u2')
        write_tiff_with_rasterio(output_paths[i], org_img) # preprocessing to tif
        org_img.tofile(output_paths2[i]) # preprocessing to yuv400

    methods = [
        'JPEG2000',
        'JPEGXL',
        'HEVC',
        'VVC',
    ]
    qs = {
        'JPEG2000': [100, 36, 32, 28, 24, 20, 16, 12, 8, 6, 4, 2],
        'JPEGXL': [100, 99.9, 99.5, 99, 98, 96, 94, 90, 85, 80, 70, 50],
        'HEVC': ['lossl', *range(-6 * (NBITS - 8), -6 * (NBITS - 8) + 54, 3)],
        'VVC': ['lossl', *range(-6 * (NBITS - 8), -6 * (NBITS - 8) + 54, 3)],
        # https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM/-/blob/master/source/Lib/CommonLib/CommonDef.h?ref_type=heads#L296
    }
    SOTA_results_dir = 'SOTA_results'
    os.makedirs(SOTA_results_dir, exist_ok=True)

    for method in methods:
        csv_file = f'{SOTA_results_dir}/test_{method}.csv'
        with open(csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            metrics = ['PSNR', 'bpsp', 'MSE', 'bits']
            
            csv_headers = ['Path'] + [f"q={q}_{metric}" for q in qs[method] for metric in metrics]
            writer.writerow(csv_headers)

            futures = []
            global MAX_WORKERS
            with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                for i, file_path in enumerate(output_paths): 
                    if method in ['HEVC', 'VVC']: file_path = output_paths2[i]
                    for q_idx, q in enumerate(qs[method]):
                        futures.append(executor.submit(run_one_rate_point, method, file_path, q, i, NBITS, C, H, W))
                        
            results = {}
            for future in as_completed(futures):
                try:
                    file_path, q, psnr, bpsp, mse_value, bits, i = future.result()
                    if file_path not in results:
                        results[file_path] = [None] * (len(qs[method]) * len(metrics))
                    q_idx = qs[method].index(q)
                    results[file_path][len(metrics) * q_idx] = psnr
                    results[file_path][len(metrics) * q_idx + 1] = bpsp
                    results[file_path][len(metrics) * q_idx + 2] = mse_value
                    results[file_path][len(metrics) * q_idx + 3] = bits
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

            for file_path in output_paths:
                row = [file_path] + results.get(file_path, [None] * (len(qs[method]) * len(metrics)))
                writer.writerow(row)


if __name__ == "__main__":
    main()
