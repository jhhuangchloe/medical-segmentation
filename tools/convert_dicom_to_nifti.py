#!/usr/bin/env python3
import argparse
import json
import os
import sys

try:
    import SimpleITK as sitk
except ImportError:
    raise ImportError('SimpleITK is required for DICOM conversion. Install it in sam2 env: pip install SimpleITK')


def main():
    parser = argparse.ArgumentParser(description='Convert a DICOM series to NIfTI and print metadata.')
    parser.add_argument('dicom_folder', help='Input folder containing DICOM files')
    parser.add_argument('output_nifti', help='Output NIfTI file path (.nii or .nii.gz)')
    parser.add_argument('--series-index', type=int, default=0,
                        help='Series index to convert if multiple DICOM series exist')
    args = parser.parse_args()

    if not os.path.isdir(args.dicom_folder):
        raise FileNotFoundError(f'DICOM folder not found: {args.dicom_folder}')

    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(args.dicom_folder)
    if not series_ids:
        raise RuntimeError(f'No DICOM series found in {args.dicom_folder}')

    if args.series_index < 0 or args.series_index >= len(series_ids):
        raise IndexError(f'Invalid series index {args.series_index}. Found {len(series_ids)} series.')

    series_id = series_ids[args.series_index]
    file_names = reader.GetGDCMSeriesFileNames(args.dicom_folder, series_id)
    if not file_names:
        raise RuntimeError('No DICOM files found for the selected series.')

    print(f'Selected series {args.series_index}: {series_id}')
    print(f'Number of files: {len(file_names)}')
    print(f'First file: {file_names[0]}')
    print(f'Last file: {file_names[-1]}')

    reader.SetFileNames(file_names)
    image = reader.Execute()

    size = image.GetSize()
    spacing = image.GetSpacing()
    origin = image.GetOrigin()
    direction = image.GetDirection()

    print('Volume size (x,y,z):', size)
    print('Spacing (x,y,z):', spacing)
    print('Origin:', origin)
    print('Direction:', direction)

    output_dir = os.path.dirname(args.output_nifti)
    if output_dir and not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    sitk.WriteImage(image, args.output_nifti)
    print('Saved NIfTI:', args.output_nifti)

    metadata = {
        'series_id': series_id,
        'num_files': len(file_names),
        'size': list(size),
        'spacing': list(spacing),
        'origin': list(origin),
        'direction': list(direction),
        'first_file': file_names[0],
        'last_file': file_names[-1],
    }
    meta_path = os.path.splitext(args.output_nifti)[0] + '.json'
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print('Saved metadata:', meta_path)


if __name__ == '__main__':
    main()
