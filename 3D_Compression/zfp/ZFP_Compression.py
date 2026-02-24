import nibabel as nib
import numpy as np
import zfpy
import os
import pickle
from typing import Tuple, Optional

def compress_nifti_with_zfp(
    input_path: str, 
    output_path: str, 
    compression_mode: str = 'rate',
    compression_param: float = 8.0,
    preserve_header: bool = True
) -> dict:
    """
    Compress NIfTI file using ZFP compression.
    
    Args:
        input_path: Path to input NIfTI file
        output_path: Path to save compressed data
        compression_mode: 'rate', 'precision', 'accuracy', or 'expert'
        compression_param: Compression parameter based on mode:
            - 'rate': bits per value (e.g., 8.0)
            - 'precision': number of bit planes (e.g., 16)
            - 'accuracy': absolute error tolerance (e.g., 1e-3)
        preserve_header: Whether to save NIfTI header separately
    
    Returns:
        Dictionary with compression statistics
    """
    
    # Load NIfTI file
    print(f"Loading NIfTI file: {input_path}")
    nii_img = nib.load(input_path)
    data = nii_img.get_fdata()
    
    # Get original data info
    original_shape = data.shape
    original_dtype = data.dtype
    original_size = data.nbytes
    
    print(f"Original data shape: {original_shape}")
    print(f"Original data type: {original_dtype}")
    print(f"Original size: {original_size / 1024 / 1024:.2f} MB")
    
    # Convert to float32 if needed (ZFP works best with float32/float64)
    if data.dtype != np.float32 and data.dtype != np.float64:
        print(f"Converting from {data.dtype} to float32")
        data = data.astype(np.float32)
    
    # Compress using ZFP
    print(f"Compressing with ZFP ({compression_mode} mode, param={compression_param})")
    
    if compression_mode == 'rate':
        compressed_data = zfpy.compress_numpy(data, rate=compression_param)
    elif compression_mode == 'precision':
        compressed_data = zfpy.compress_numpy(data, precision=int(compression_param))
    elif compression_mode == 'accuracy':
        compressed_data = zfpy.compress_numpy(data, tolerance=compression_param)
    else:
        raise ValueError("compression_mode must be 'rate', 'precision', or 'accuracy'")
    
    compressed_size = len(compressed_data)
    compression_ratio = original_size / compressed_size
    
    # Prepare data to save
    save_data = {
        'compressed_data': compressed_data,
        'original_shape': original_shape,
        'original_dtype': str(original_dtype),
        'compression_mode': compression_mode,
        'compression_param': compression_param
    }
    
    # Save NIfTI header if requested
    if preserve_header:
        save_data['nifti_header'] = nii_img.header
        save_data['nifti_affine'] = nii_img.affine
    
    # Save compressed data
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    # Print compression statistics
    print(f"Compressed size: {compressed_size / 1024 / 1024:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}:1")
    print(f"Space saved: {(1 - 1/compression_ratio) * 100:.1f}%")
    
    return {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': compression_ratio,
        'space_saved_percent': (1 - 1/compression_ratio) * 100
    }

def decompress_nifti_from_zfp(
    compressed_path: str, 
    output_path: str,
    output_format: str = 'nifti'
) -> None:
    """
    Decompress ZFP-compressed NIfTI data.
    
    Args:
        compressed_path: Path to compressed data file
        output_path: Path to save decompressed NIfTI file
        output_format: 'nifti' or 'numpy' - output format
    """
    
    # Load compressed data
    print(f"Loading compressed data: {compressed_path}")
    with open(compressed_path, 'rb') as f:
        save_data = pickle.load(f)
    
    # Extract information
    compressed_data = save_data['compressed_data']
    original_shape = save_data['original_shape']
    original_dtype = save_data['original_dtype']
    
    # Decompress
    print("Decompressing data...")
    decompressed_data = zfpy.decompress_numpy(compressed_data)
    
    # Reshape to original dimensions
    decompressed_data = decompressed_data.reshape(original_shape)
    
    # Convert back to original dtype if needed
    if original_dtype != 'float32' and original_dtype != 'float64':
        decompressed_data = decompressed_data.astype(original_dtype)
    
    if output_format == 'nifti':
        # Create NIfTI image
        if 'nifti_header' in save_data and 'nifti_affine' in save_data:
            # Use preserved header and affine
            nii_img = nib.Nifti1Image(decompressed_data, 
                                     save_data['nifti_affine'], 
                                     save_data['nifti_header'])
        else:
            # Create basic NIfTI image
            nii_img = nib.Nifti1Image(decompressed_data, np.eye(4))
        
        # Save NIfTI file
        nib.save(nii_img, output_path)
        print(f"Decompressed NIfTI saved to: {output_path}")
        
    elif output_format == 'numpy':
        # Save as numpy array
        np.save(output_path, decompressed_data)
        print(f"Decompressed numpy array saved to: {output_path}")
    
    print(f"Decompressed data shape: {decompressed_data.shape}")

def get_base_filename(filename: str) -> str:
    """
    Extract base filename from NIfTI file, handling both .nii and .nii.gz extensions.
    
    Args:
        filename: Original filename (e.g., 'file.nii' or 'file.nii.gz')
    
    Returns:
        Base filename without extension (e.g., 'file')
    """
    if filename.endswith('.nii.gz'):
        return filename[:-7]  # Remove '.nii.gz'
    elif filename.endswith('.nii'):
        return filename[:-4]  # Remove '.nii'
    else:
        # Fallback for other extensions
        return os.path.splitext(filename)[0]

def compress_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    compression_mode: str = 'rate',
    compression_param: float = 8.0,
    preserve_header: bool = True
) -> None:
    """
    Compress all NIfTI files in a directory using ZFP compression.
    
    Args:
        input_dir: Path to directory containing NIfTI files
        output_dir: Path to output directory (if None, uses input_dir)
        compression_mode: 'rate', 'precision', or 'accuracy'
        compression_param: Compression parameter
        preserve_header: Whether to preserve NIfTI headers
    """
    
    # Use input directory as output directory if not specified
    if output_dir is None:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all NIfTI files
    nifti_extensions = ['.nii', '.nii.gz']
    nifti_files = []
    
    for filename in os.listdir(input_dir):
        if any(filename.endswith(ext) for ext in nifti_extensions):
            nifti_files.append(filename)
    
    if not nifti_files:
        print(f"No NIfTI files found in {input_dir}")
        return
    
    print(f"Found {len(nifti_files)} NIfTI files in {input_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    total_original_size = 0
    total_compressed_size = 0
    successful_compressions = 0
    
    for i, filename in enumerate(nifti_files, 1):
        input_path = os.path.join(input_dir, filename)
        
        # Generate output filename: same base name but with .zfp extension
        base_name = get_base_filename(filename)
        output_filename = f"{base_name}.zfp"
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"[{i}/{len(nifti_files)}] Processing: {filename}")
        print(f"Output: {output_filename}")
        
        try:
            stats = compress_nifti_with_zfp(
                input_path, 
                output_path, 
                compression_mode, 
                compression_param,
                preserve_header
            )
            
            total_original_size += stats['original_size']
            total_compressed_size += stats['compressed_size']
            successful_compressions += 1
            
        except Exception as e:
            print(f"ERROR processing {filename}: {str(e)}")
            continue
        
        print("-" * 40)
    
    # Print overall statistics
    if successful_compressions > 0:
        overall_ratio = total_original_size / total_compressed_size
        print(f"\n{'='*20} COMPRESSION SUMMARY {'='*20}")
        print(f"Files processed: {successful_compressions}/{len(nifti_files)}")
        print(f"Total original size: {total_original_size / 1024 / 1024:.2f} MB")
        print(f"Total compressed size: {total_compressed_size / 1024 / 1024:.2f} MB")
        print(f"Overall compression ratio: {overall_ratio:.2f}:1")
        print(f"Total space saved: {(1 - 1/overall_ratio) * 100:.1f}%")
        print("=" * 59)
    else:
        print("No files were successfully compressed.")

def decompress_directory(
    input_dir: str,
    output_dir: Optional[str] = None,
    output_format: str = 'nifti'
) -> None:
    """
    Decompress all .zfp files in a directory back to NIfTI format.
    
    Args:
        input_dir: Path to directory containing .zfp files
        output_dir: Path to output directory (if None, uses input_dir)
        output_format: 'nifti' or 'numpy'
    """
    
    # Use input directory as output directory if not specified
    if output_dir is None:
        output_dir = input_dir
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Find all .zfp files
    zfp_files = [f for f in os.listdir(input_dir) if f.endswith('.zfp')]
    
    if not zfp_files:
        print(f"No .zfp files found in {input_dir}")
        return
    
    print(f"Found {len(zfp_files)} .zfp files in {input_dir}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    successful_decompressions = 0
    
    for i, filename in enumerate(zfp_files, 1):
        input_path = os.path.join(input_dir, filename)
        
        # Generate output filename
        base_name = filename[:-4]  # Remove .zfp extension
        if output_format == 'nifti':
            output_filename = f"{base_name}.nii"
        else:
            output_filename = f"{base_name}.npy"
        
        output_path = os.path.join(output_dir, output_filename)
        
        print(f"[{i}/{len(zfp_files)}] Decompressing: {filename}")
        print(f"Output: {output_filename}")
        
        try:
            decompress_nifti_from_zfp(input_path, output_path, output_format)
            successful_decompressions += 1
            
        except Exception as e:
            print(f"ERROR decompressing {filename}: {str(e)}")
            continue
        
        print("-" * 40)
    
    print(f"\nDecompression complete: {successful_decompressions}/{len(zfp_files)} files processed successfully.")

def analyze_nifti_file(input_path: str) -> None:
    """
    Analyze a NIfTI file to understand size discrepancies.
    """
    print("=== NIfTI FILE ANALYSIS ===")
    
    # File size on disk
    disk_size = os.path.getsize(input_path)
    print(f"File size on disk: {disk_size / 1024 / 1024:.2f} MB")
    
    # Check if compressed
    is_compressed = input_path.endswith('.gz')
    print(f"File is gzip compressed: {is_compressed}")
    
    # Load and analyze
    nii_img = nib.load(input_path)
    header = nii_img.header
    
    # Get info from header (before loading data)
    shape = header.get_data_shape()
    disk_dtype = header.get_data_dtype()
    voxel_count = np.prod(shape)
    
    print(f"\nHeader Information:")
    print(f"  Shape: {shape}")
    print(f"  Data type on disk: {disk_dtype}")
    print(f"  Voxels: {voxel_count:,}")
    print(f"  Bytes per voxel: {np.dtype(disk_dtype).itemsize}")
    
    # Calculate theoretical sizes
    theoretical_uncompressed = voxel_count * np.dtype(disk_dtype).itemsize
    print(f"  Theoretical uncompressed size: {theoretical_uncompressed / 1024 / 1024:.2f} MB")
    
    if is_compressed:
        compression_ratio = theoretical_uncompressed / disk_size
        print(f"  Gzip compression ratio: {compression_ratio:.2f}:1")
    
    # Now load the actual data
    print(f"\nLoading data into memory...")
    data = nii_img.get_fdata()
    
    print(f"Loaded data information:")
    print(f"  Shape: {data.shape}")
    print(f"  Data type in memory: {data.dtype}")
    print(f"  Size in memory: {data.nbytes / 1024 / 1024:.2f} MB")
    
    # Data range analysis
    print(f"\nData range analysis:")
    print(f"  Min value: {np.min(data):.6f}")
    print(f"  Max value: {np.max(data):.6f}")
    print(f"  Mean: {np.mean(data):.6f}")
    print(f"  Std: {np.std(data):.6f}")
    
    # Check for data type efficiency
    data_min, data_max = np.min(data), np.max(data)
    
    if data.dtype == np.float64:
        print(f"\nNote: Data is float64. If values fit in float32 range, you could save 50% memory.")
        if data_min >= np.finfo(np.float32).min and data_max <= np.finfo(np.float32).max:
            print("  → Values fit in float32 range")
        else:
            print("  → Values require float64 precision")
    
    print("=" * 50)
