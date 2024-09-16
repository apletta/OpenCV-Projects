import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

"""
Implementation from blog tutorial: https://learnopencv.com/exposure-fusion-using-opencv-cpp-python/
"""

def mergeMertens(images):
  # Merge
  mergeMertens = cv2.createMergeMertens()
  result = mergeMertens.process(images)
  
  # Normalize to prevent clipping and convert to integer type
  result = cv2.normalize(result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

  return result  

def alignMTB(images):
  # Assumes all images have the same channel depth, either 1 or 3
  images_3chan = images
  if len(images[0].shape) == 2:
    images_3chan = [cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) for image in images]

  # Do alignment
  alignMTB = cv2.createAlignMTB()
  alignMTB.process(images_3chan, images_3chan)

  # Convert back if necessary
  images_aligned = images_3chan
  if len(images[0].shape) == 2:
    images_aligned = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) for image in images_3chan]

  # Return result
  return images_aligned


if __name__ == '__main__':
  # Read in images
  folder_dir = 'st_louis'
  # folder_dir = 'memorial_church'

  src_dir = Path(__file__).resolve().parent
  images_dir = src_dir / '..' / 'images' / folder_dir

  if folder_dir == 'memorial_church':
    image_filenames = sorted(Path(images_dir).glob('*.png'))
  else: # folder_dir == 'st_louis'
    image_filenames = sorted(Path(images_dir).glob('*.jpeg'))
  
  images_rgb = []
  images_gray = []
  image_filenames_str = []
  for image_filename in image_filenames:
    image_filenames_str.append(image_filename.name)
    # Read in grayscale image
    image_gray = cv2.imread(image_filename.as_posix(), cv2.IMREAD_GRAYSCALE)
    images_gray.append(image_gray)

    image_rgb = cv2.cvtColor(cv2.imread(image_filename.as_posix()), cv2.COLOR_BGR2RGB)
    images_rgb.append(image_rgb)

  # Align input images
  images_rgb_aligned = alignMTB(images_rgb)
  images_gray_aligned = alignMTB(images_gray)

  # Merge
  result_rgb = mergeMertens(images_rgb_aligned)
  result_gray = mergeMertens(images_gray_aligned)
  
  # Check rgb vs gray
  result_rgb_to_gray = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2GRAY)
  result_diff = np.absolute(result_rgb_to_gray.astype(np.float32) - result_gray.astype(np.float32)).astype(np.uint8)

  # Get sift features
  sift = cv2.SIFT_create()
  kp_rgb = sift.detect(result_rgb, None)
  kp_rgb_to_gray = sift.detect(result_rgb_to_gray, None)
  kp_gray = sift.detect(result_gray, None)
  kp_diff = sift.detect(result_diff, None)

  
  result_rgb_kp = cv2.drawKeypoints(result_rgb, kp_rgb, 0, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
  result_gray_kp = cv2.drawKeypoints(result_gray, kp_gray, 0, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
  result_rgb_to_gray_kp = cv2.drawKeypoints(result_rgb_to_gray, kp_rgb_to_gray, 0, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
  result_diff_kp = cv2.drawKeypoints(result_diff, kp_diff, 0, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

  # Display
  fig, ax = plt.subplots(2,4, figsize=(14, 8))
  fig.suptitle("Exposure Fusion (AEB/HDR)")

  for i in range(len(images_rgb)):
    ax[0,i].imshow(images_rgb[i])
    ax[0,i].set_title(image_filenames_str[i])

  ax[1,0].imshow(result_rgb_kp)
  ax[1,0].set_title(f"RGB Input\nN_sift = {len(kp_rgb)}")
  
  ax[1,1].imshow(result_rgb_to_gray_kp, cmap='gray', vmin=0, vmax=255)
  ax[1,1].set_title(f"RGB Input Converted to Grayscale\nN_sift = {len(kp_rgb_to_gray)}")
  
  ax[1,2].imshow(result_gray_kp, cmap='gray', vmin=0, vmax=255)
  ax[1,2].set_title(f"Grayscale Input\nN_sift = {len(kp_gray)}")
  
  ax[1,3].imshow(result_diff_kp, cmap='gray', vmin=0, vmax=255)
  ax[1,3].set_title(f"RGB Input vs. Grayscale Input\nN_sift = {len(kp_diff)}")
  
  plt.tight_layout()
  plt.show()
