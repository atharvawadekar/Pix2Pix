import os
import re
from collections import defaultdict

def sorted_alphanumeric(data):
    """Sort filenames properly."""
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

def analyze_cuhk_dataset(data_root):
    """Analyze CUHK dataset structure and find pairing issues."""
    
    image_path = os.path.join(data_root, "photos")
    sketch_path = os.path.join(data_root, "sketches")
    
    print("=== CUHK Dataset Analysis ===")
    print(f"Dataset root: {data_root}")
    print(f"Photos path: {image_path}")
    print(f"Sketches path: {sketch_path}")
    print()
    
    # Check if directories exist
    if not os.path.exists(image_path):
        print(f"ERROR: Photos directory not found: {image_path}")
        return
    if not os.path.exists(sketch_path):
        print(f"ERROR: Sketches directory not found: {sketch_path}")
        return
    
    # Get file lists
    image_files = sorted_alphanumeric(os.listdir(image_path))
    sketch_files = sorted_alphanumeric(os.listdir(sketch_path))
    
    print(f"File Counts:")
    print(f"  Photos: {len(image_files)}")
    print(f"  Sketches: {len(sketch_files)}")
    print()
    
    # Show first few files
    print("Sample Files:")
    print("  Photos (first 10):")
    for i, f in enumerate(image_files[:10]):
        print(f"    {i+1:2d}. {f}")
    
    print("  Sketches (first 10):")
    for i, f in enumerate(sketch_files[:10]):
        print(f"    {i+1:2d}. {f}")
    print()
    
    # Analyze filename patterns
    print("Filename Analysis:")
    
    # Extract base names
    image_bases = []
    sketch_bases = []
    
    for img_file in image_files:
        base = os.path.splitext(img_file)[0]
        base = base.replace('_photo', '').replace('-photo', '').replace('_real', '')
        image_bases.append(base)
    
    for sketch_file in sketch_files:
        base = os.path.splitext(sketch_file)[0]
        base = base.replace('_sketch', '').replace('-sketch', '').replace('_drawing', '')
        sketch_bases.append(base)
    
    # Find matches
    image_set = set(image_bases)
    sketch_set = set(sketch_bases)
    
    matches = image_set.intersection(sketch_set)
    image_only = image_set - sketch_set
    sketch_only = sketch_set - image_set
    
    print(f"  Matched pairs: {len(matches)}")
    print(f"  Photos without sketches: {len(image_only)}")
    print(f"  Sketches without photos: {len(sketch_only)}")
    print()
    
    if len(matches) > 0:
        print("Found matching pairs (first 10):")
        for i, base in enumerate(sorted(list(matches))[:10]):
            # Find original filenames
            img_file = None
            sketch_file = None
            
            for f in image_files:
                if base in os.path.splitext(f)[0]:
                    img_file = f
                    break
            
            for f in sketch_files:
                if base in os.path.splitext(f)[0]:
                    sketch_file = f
                    break
            
            print(f"    {i+1:2d}. {base}: {img_file} <-> {sketch_file}")
    
    if len(image_only) > 0:
        print(f"\nPhotos without matching sketches (first 10):")
        for i, base in enumerate(sorted(list(image_only))[:10]):
            print(f"    {i+1:2d}. {base}")
    
    if len(sketch_only) > 0:
        print(f"\nSketches without matching photos (first 10):")
        for i, base in enumerate(sorted(list(sketch_only))[:10]):
            print(f"    {i+1:2d}. {base}")
    
    # Recommendations
    print(f"\nRecommendations:")
    if len(matches) == 0:
        print("  ERROR: No filename-based matches found!")
        print("  TIP: Try positional pairing (pair by file order)")
        print("  TIP: Check if files have different naming conventions")
    elif len(matches) < min(len(image_files), len(sketch_files)):
        print(f"  WARNING: Only {len(matches)} pairs found out of {min(len(image_files), len(sketch_files))} possible")
        print("  TIP: Some files don't follow the expected naming pattern")
    else:
        print(f"  GOOD: Found {len(matches)} valid pairs")
    
    print(f"\nSuggested fixes:")
    print(f"  1. Use the updated load_cuhk_dataset() function")
    print(f"  2. The function will now find {len(matches)} valid pairs")
    print(f"  3. After 8x augmentation: {len(matches) * 8} training samples")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python debug_dataset.py <path_to_dataset>")
        sys.exit(1)
    
    data_root = sys.argv[1]
    analyze_cuhk_dataset(data_root)