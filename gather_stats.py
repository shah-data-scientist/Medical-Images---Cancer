"""
Quick script to gather dataset statistics for updating notebook markdown.
"""
import os
from pathlib import Path
from PIL import Image
import json

def gather_dataset_stats():
    """Gather comprehensive statistics about the dataset."""

    # Define paths
    DATA_DIR = Path('data')
    LABELLED_DIR = DATA_DIR / 'labelled'
    UNLABELLED_DIR = DATA_DIR / 'unlabelled'

    stats = {
        'dataset_overview': {},
        'image_properties': {},
        'file_sizes': {},
        'observations': []
    }

    # Count images
    try:
        cancer_images = list((LABELLED_DIR / 'cancer').glob('*.jpg'))
        normal_images = list((LABELLED_DIR / 'normal').glob('*.jpg'))
        unlabelled_images = list(UNLABELLED_DIR.glob('*.jpg'))

        stats['dataset_overview'] = {
            'total_images': len(cancer_images) + len(normal_images) + len(unlabelled_images),
            'labeled_cancer': len(cancer_images),
            'labeled_normal': len(normal_images),
            'unlabeled': len(unlabelled_images),
            'class_balance_ratio': round(len(cancer_images) / len(normal_images), 2) if len(normal_images) > 0 else 0
        }

        # Sample image properties (try to open a few images)
        sample_images = []
        if cancer_images:
            sample_images.append(('cancer', cancer_images[0]))
        if normal_images:
            sample_images.append(('normal', normal_images[0]))
        if unlabelled_images:
            sample_images.append(('unlabeled', unlabelled_images[0]))

        dimensions = set()
        modes = set()
        file_sizes = []

        for category, img_path in sample_images:
            try:
                # Get file size
                file_size_kb = os.path.getsize(img_path) / 1024
                file_sizes.append(file_size_kb)

                # Open and check properties
                with Image.open(img_path) as img:
                    dimensions.add(f"{img.size[0]}×{img.size[1]}")
                    modes.add(img.mode)

                    stats['image_properties'][category] = {
                        'dimensions': f"{img.size[0]}×{img.size[1]}",
                        'mode': img.mode,
                        'channels': len(img.getbands()),
                        'format': img.format,
                        'file_size_kb': round(file_size_kb, 2)
                    }
            except Exception as e:
                print(f"Error reading {category} image: {e}")
                stats['image_properties'][category] = {'error': str(e)}

        # Overall properties
        stats['file_sizes'] = {
            'avg_kb': round(sum(file_sizes) / len(file_sizes), 2) if file_sizes else 0,
            'min_kb': round(min(file_sizes), 2) if file_sizes else 0,
            'max_kb': round(max(file_sizes), 2) if file_sizes else 0
        }

        stats['observations'].append(f"Dataset contains {stats['dataset_overview']['total_images']} total images")
        stats['observations'].append(f"Class balance: {stats['dataset_overview']['class_balance_ratio']:.2f} (cancer/normal ratio)")

        if len(dimensions) == 1:
            stats['observations'].append(f"All images have consistent dimensions: {list(dimensions)[0]}")
        else:
            stats['observations'].append(f"Multiple image dimensions found: {dimensions}")

        if len(modes) == 1:
            stats['observations'].append(f"All images have the same mode: {list(modes)[0]}")
        else:
            stats['observations'].append(f"Multiple image modes found: {modes}")

    except Exception as e:
        stats['error'] = str(e)

    return stats

if __name__ == "__main__":
    print("Gathering dataset statistics...\n")
    stats = gather_dataset_stats()

    # Print stats
    print("="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(json.dumps(stats, indent=2))

    # Save to file
    with open('dataset_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n✓ Statistics saved to dataset_stats.json")
