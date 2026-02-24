import os
import sys
from pathlib import Path

from compression import compress_nifti_with_zfp, get_base_filename

def create_zfp_folders(input_dir: str):
    """
    Create _zfp versions of all folders containing NIfTI files.
    Skips JSON files in main directory.
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"Error: Directory {input_dir} does not exist")
        return
    
    # Get all subdirectories
    folders = [d for d in input_path.iterdir() if d.is_dir()]
    
    if not folders:
        print("No folders found in directory")
        return
    
    print(f"Processing {len(folders)} folders...")
    
    total_files = 0
    successful = 0
    
    for folder in folders:
        print(f"\nProcessing folder: {folder.name}")
        
        # Find NIfTI files in this folder (recursively)
        nifti_files = []
        for ext in ['.nii', '.nii.gz']:
            nifti_files.extend(folder.rglob(f'*{ext}'))
        
        if not nifti_files:
            print(f"  No NIfTI files found in {folder.name}")
            continue
        
        # Create _zfp folder
        zfp_folder = input_path / f"{folder.name}_zfp"
        print(f"  Found {len(nifti_files)} NIfTI files")
        print(f"  Creating: {zfp_folder.name}")
        
        for nifti_file in nifti_files:
            # Get relative path from original folder
            rel_path = nifti_file.relative_to(folder)
            
            # Create output path in _zfp folder
            base_name = get_base_filename(nifti_file.name)
            output_file = zfp_folder / rel_path.parent / f"{base_name}.zfp"
            
            # Create output directory
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                compress_nifti_with_zfp(
                    str(nifti_file), 
                    str(output_file),
                    compression_mode='rate',
                    compression_param=8.0
                )
                successful += 1
                print(f"    ✓ {rel_path} -> {base_name}.zfp")
                
            except Exception as e:
                print(f"    ✗ Failed {rel_path}: {e}")
            
            total_files += 1
    
    print(f"\nDone! Processed {successful}/{total_files} files")

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <directory_path>")
        print("Example: python main.py /path/to/data")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    create_zfp_folders(input_dir)

if __name__ == "__main__":
    main()