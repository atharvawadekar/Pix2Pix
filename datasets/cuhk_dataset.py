import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import re

def sorted_alphanumeric(data):
    #bring non numeric to lower case and convert numeric to digits for sorting.
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    #split name into non-numric and numeric
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

    #sort the filenames based on our defined function
    return sorted(data, key=alphanum_key)

def load_and_preprocess_image(filepath, size=256):
    image = cv2.imread(filepath)
    if image is None:
        print(f"Filepath incorrect!!!")
        return None
    #change to rgb as imread loads in bgr
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #resize
    image = cv2.resize(image, (size, size))
    
    return image

def augment_image(image):
    #types of augmenttions- original, hori flip, vert flip, vert and hori flip, cw 90, cw 90 hori, ccw 90, ccw 90 hori
    aug_list = [
        image,
        cv2.flip(image, 1),
        cv2.flip(image, 0),
        cv2.flip(cv2.flip(image, 0), 1),
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        cv2.flip(cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), 1),
        cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE),
        cv2.flip(cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
    ]
    #normalize to -1 and 1
    #divide by 127.5 makes values in [0-2] and subtracting 1 shifts it in [-1,1]
    return [(img.astype('float32') / 127.5) - 1.0 for img in aug_list]

def augment_sketch(sketch):
        sketch_augs = [
            sketch,
            cv2.flip(sketch, 1),
            cv2.flip(sketch, 0),
            cv2.flip(sketch, -1),
            cv2.rotate(sketch, cv2.ROTATE_90_CLOCKWISE),
            cv2.flip(cv2.rotate(sketch, cv2.ROTATE_90_CLOCKWISE), 1),
            cv2.rotate(sketch, cv2.ROTATE_90_COUNTERCLOCKWISE),
            cv2.flip(cv2.rotate(sketch, cv2.ROTATE_90_COUNTERCLOCKWISE), 1)
        ]
        
        return [(s.astype('float32') / 127.5) - 1.0 for s in sketch_augs]

def load_cuhk_dataset(data_root, size=256):
    #real photos in photos folder
    image_path = os.path.join(data_root, "photos")
    #sketches in sketches folder
    sketch_path = os.path.join(data_root, "sketches")
    
    #check if paths exist
    if not os.path.exists(image_path) or not os.path.exists(sketch_path):
        raise ValueError(f"Could not find 'photos' and 'sketches' folders in {data_root}")
    
    #Needs to be improved but working as of now luckily ---- NEED TO THINK
    image_files = sorted_alphanumeric(os.listdir(image_path))
    sketch_files = sorted_alphanumeric(os.listdir(sketch_path))
    
    #Verify if the number for images and sketches is the same
    print(f"Found {len(image_files)} real images.")
    print(f"Found {len(sketch_files)} sketches.")
    
    #fhinding matching pairs of images and sketches
    valid_pairs = []
    
    #create a key for every image path - this doesnt work due to weird naming
    # image_dict = {}
    # for img_file in image_files:
    #     #The base name is in images and sketches
    #     base_name = os.path.splitext(img_file)[0]
    #     base_name = base_name.replace('_photo', '').replace('-photo', '').replace('_real', '')
    #     image_dict[base_name] = img_file
    
    # sketch_dict = {}
    # for sketch_file in sketch_files:
    #     base_name = os.path.splitext(sketch_file)[0]
    #     base_name = base_name.replace('_sketch', '').replace('-sketch', '').replace('_drawing', '')
    #     sketch_dict[base_name] = sketch_file
    
    # # Find matching pairs
    # for base_name in image_dict:
    #     if base_name in sketch_dict:
    #         valid_pairs.append((image_dict[base_name], sketch_dict[base_name]))
    
    # print(f"Found {len(valid_pairs)} valid sketch-photo pairs")
    

    #need to think here- works with a bit of luck
    print("No name-based matches found, trying positional pairing...")
    min_count = min(len(image_files), len(sketch_files))
    valid_pairs = list(zip(image_files[:min_count], sketch_files[:min_count]))
    print(f"Created {len(valid_pairs)} positional pairs")
    
    img_array = []
    sketch_array = []
    
    #processing the found valid pairs
    for img_file, sketch_file in tqdm(valid_pairs, desc="LLoad and augment images - progress bar:"):
        #load image
        img_fp = os.path.join(image_path, img_file)
        image = load_and_preprocess_image(img_fp, size=size)
        if image is None:
            continue
        
        #load sketch
        sketch_fp = os.path.join(sketch_path, sketch_file)
        sketch = load_and_preprocess_image(sketch_fp, size=size)
        if sketch is None:
            continue
        
        #sketches should be in grayscale
        if len(sketch.shape) == 3:
            sketch = cv2.cvtColor(sketch, cv2.COLOR_RGB2GRAY)
        
        #apply augmentations
        img_augs = augment_image(image)
        
        sketch_normalized = augment_sketch(sketch)
        
        #append to arrays
        img_array.extend(img_augs)
        sketch_array.extend(sketch_normalized)
    
    #convert aur images to grayscale
    img_array_gray = []
    for img in img_array:
        if len(img.shape) == 3: 
            #covert to 0-255 range again
            img_uint8 = ((img + 1.0) * 127.5).astype(np.uint8)
            #rgb to gray conversion
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            #normalize our grayscale image
            gray_normalized = (gray.astype('float32') / 127.5) - 1.0
            #append
            img_array_gray.append(gray_normalized)
        else:
            img_array_gray.append(img)
    
    print(f"Total number of sketch images: {len(sketch_array)}")
    print(f"Total number of real images: {len(img_array_gray)}")
    
    # Ensure equal number of sketches and photos
    if len(img_array_gray) != len(sketch_array):
        print(f"Mismatched")

        #Need a better way to handle mismatched- we know we will always have matched for this dataset. Maybe just stop if mismatched
        min_len = min(len(img_array_gray), len(sketch_array))
        img_array_gray = img_array_gray[:min_len]
        sketch_array = sketch_array[:min_len]

        print(f"Final dataset size: {min_len} pairs")
    #return sketch and images both grayscale and normalized
    return np.array(sketch_array), np.array(img_array_gray)


# def save_preprocessed_data(sketches, photos, output_dir='./preprocessed_data'):
#     os.makedirs(output_dir, exist_ok=True)
    
#     sketch_path = os.path.join(output_dir, 'sketch_images.npy')
#     photo_path = os.path.join(output_dir, 'real_images.npy')
    
#     np.save(sketch_path, sketches)
#     np.save(photo_path, photos)
    
#     print(f"Preprocessed data saved to:")
#     print(f"  Sketches: {sketch_path}")
#     print(f"  Photos: {photo_path}")

# def load_preprocessed_data(data_dir='./preprocessed_data'):
#     """Load previously saved preprocessed data.
    
#     Args:
#         data_dir: Directory containing saved arrays
        
#     Returns:
#         Tuple of (sketches, photos) as numpy arrays
#     """
#     sketch_path = os.path.join(data_dir, 'sketch_images.npy')
#     photo_path = os.path.join(data_dir, 'real_images.npy')
    
#     if not (os.path.exists(sketch_path) and os.path.exists(photo_path)):
#         raise FileNotFoundError(f"Preprocessed data not found in {data_dir}")
    
#     sketches = np.load(sketch_path)
#     photos = np.load(photo_path)
    
#     print(f"Loaded preprocessed data:")
#     print(f"  Sketches: {sketches.shape}")
#     print(f"  Photos: {photos.shape}")
    
#     return sketches, photos

class CUHKFaceSketchDataset(Dataset):    
    def __init__(self, sketches, photos, mode='train', train_ratio=0.8, val_ratio=0.1):
        self.mode = mode
        
        # Calculate split indices
        total_pairs = len(sketches)
        #get train split
        train_split = int(train_ratio * total_pairs)
        #get val split
        val_split = int((train_ratio + val_ratio) * total_pairs)
        #rest will be test split
        
        #split based on mode
        if mode == 'train':
            self.sketches = sketches[:train_split]
            self.photos = photos[:train_split]
        elif mode == 'val':
            self.sketches = sketches[train_split:val_split]
            self.photos = photos[train_split:val_split]
        elif mode == 'test':
            self.sketches = sketches[val_split:]
            self.photos = photos[val_split:]
        else:
            raise ValueError(f"Invalid mode: {mode}. Use 'train', 'val', or 'test'")
        
        print(f"Loaded {len(self.sketches)} {mode} pairs")

    def __len__(self):
        #return number of sketches
        return len(self.sketches)

    def __getitem__(self, idx):
        #return at value for the index
        sketch = torch.from_numpy(self.sketches[idx]).unsqueeze(0).float()
        photo = torch.from_numpy(self.photos[idx]).unsqueeze(0).float()
        return sketch, photo

def get_dataset_statistics(sketches, photos):
    print("\n=== Dataset Statistics ===")
    print(f"Total pairs: {len(sketches)}")
    print(f"Sketch shape: {sketches.shape}")
    print(f"Photo shape: {photos.shape}")
    print(f"Sketch range: [{sketches.min():.3f}, {sketches.max():.3f}]")
    print(f"Photo range: [{photos.min():.3f}, {photos.max():.3f}]")
    print(f"Sketch mean: {sketches.mean():.3f}, std: {sketches.std():.3f}")
    print(f"Photo mean: {photos.mean():.3f}, std: {photos.std():.3f}")