import SimpleITK as sitk  # For loading the dataset
import numpy as np  # For data manipulation
import glob  # For populating the list of files
from scipy.ndimage import zoom  # For resizing
import re  # For parsing the filenames (to know their modality)
import math



def read_img(img_path):
    """
    Reads a .nii.gz image and returns as a numpy array.
    """
    return sitk.GetArrayFromImage(sitk.ReadImage(img_path))


def resize(img, shape, mode='constant', orig_shape=(155, 240, 240)):
    """
    Wrapper for scipy.ndimage.zoom suited for MRI images.
    """
    assert len(shape) == 3, "Can not have more than 3 dimensions"
    factors = (
        shape[0] / orig_shape[0],
        shape[1] / orig_shape[1],
        shape[2] / orig_shape[2]
    )

    # Resize to the given shape
    return zoom(img, factors, mode=mode)


def preprocess(img, out_shape=None):
    """
    Preprocess the image.
    Just an example, you can add more preprocessing steps if you wish to.
    """
    if out_shape is not None:
        img = resize(img, out_shape, mode='constant')

    # Normalize the image
    mean = img.mean()
    std = img.std()
    return (img - mean) / std


def preprocess_label(img, out_shape=None, mode='nearest'):
    """
    Separates out the 3 labels from the segmentation provided, namely:
    GD-enhancing tumor (ET — label 4), the peritumoral edema (ED — label 2))
    and the necrotic and non-enhancing tumor core (NCR/NET — label 1)
    """
    ncr = img == 1  # Necrotic and Non-Enhancing Tumor (NCR/NET)
    ed = img == 2  # Peritumoral Edema (ED)
    et = img == 4  # GD-enhancing Tumor (ET)

    if out_shape is not None:
        ncr = resize(ncr, out_shape, mode=mode)
        ed = resize(ed, out_shape, mode=mode)
        et = resize(et, out_shape, mode=mode)

    return np.array([ncr, ed, et], dtype=np.uint8)

def get_Datas():
    # Get a list of files for all modalities individually
    t1 = glob.glob('images/*GG/*/*t1.nii.gz')
    t2 = glob.glob('images/*GG/*/*t2.nii.gz')
    flair = glob.glob('images/*GG/*/*flair.nii.gz')
    t1ce = glob.glob('images/*GG/*/*t1ce.nii.gz')
    seg = glob.glob('images/*GG/*/*seg.nii.gz')  # Ground Truth


    pat = re.compile('.*_(\w*)\.nii\.gz')

    data_paths = [{
        pat.findall(item)[0]:item
        for item in items
    }
    for items in list(zip(t1, t2, t1ce, flair, seg))]

    input_shape = (4, 80, 96, 64)
    output_channels = 3
    data = np.empty((len(data_paths[:4]),) + input_shape, dtype=np.float32)
    labels = np.empty((len(data_paths[:4]), output_channels) + input_shape[1:], dtype=np.uint8)

    # Parameters for the progress bar
    total = len(data_paths[:4])
    step = 25 / total

    for i, imgs in enumerate(data_paths[:4]):
        try:
            data[i] = np.array([preprocess(read_img(imgs[m]), input_shape[1:]) for m in ['t1', 't2', 't1ce', 'flair']],
                               dtype=np.float32)
            labels[i] = preprocess_label(read_img(imgs['seg']), input_shape[1:])[None, ...]

            # Print the progress bar
            print('\r' + f'Progress: '
                         f"[{'=' * int((i + 1) * step) + ' ' * (24 - int((i + 1) * step))}]"
                         f"({math.ceil((i + 1) * 100 / (total))} %)",
                  end='')
        except Exception as e:
            print(f'Something went wrong with {imgs["t1"]}, skipping...\n Exception:\n{str(e)}')
            continue
    return data, labels, input_shape

