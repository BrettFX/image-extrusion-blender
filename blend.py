"""
Author: Brett Allen
Created: 10/02/2022
"""
import cv2
from matplotlib import pyplot as plt
import argparse
import logging
import os

__description__ = """Image extrusion script to combine two images together and elevate/extruded
edges of the second image, blended with the first image. This process
makes use of the Sobel edge detection algorithm along with a weighted
sum blending algorithm to achieve the desired result."""

parser = argparse.ArgumentParser(description=__description__, prog="Python Image Extrusion Blender")
parser.add_argument('background', help='Input background image.')
parser.add_argument('foreground', help='Input foreground image.')
parser.add_argument('--sobel-x', '-x', dest='use_sobel_x', action='store_true', default=True, help='Use derivative x value for Sobel calculation. Default True.')
parser.add_argument('--sobel-y', '-y', dest='use_sobel_y', action='store_true', default=False, help='Use derivative y value for Sobel calculation. Default False.')
parser.add_argument('--outpath', '-o', default='blended.png', help='Path write blended file to. Default "blended.png".')
parser.add_argument('--debug', '-d', dest='debug_mode', action='store_true', default=False, help='Run program in debug mode to show verbose logging and display intermediate steps. Default False.')
parser.add_argument('--version', '-v', action='version', version='%(prog)s Version 1.0')

args = parser.parse_args()
DEBUG_MODE = args.debug_mode

logging.basicConfig(
  level=logging.INFO if not DEBUG_MODE else logging.DEBUG,
  format='%(asctime)s {%(filename)s:%(lineno)d} [%(levelname)s] %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S',
)

logger = logging.getLogger("image-extrusion-blender")

def display(img, title="Image", cmap="gray"):
  plt.figure()
  plt.grid(False)
  plt.axis('off')
  plt.imshow(img, cmap=cmap)
  plt.title(title), plt.xticks([]), plt.yticks([])
  plt.show(block=True)

def display_all(images: list, column_limit=2, cmap="gray") -> None:
  rows = 1
  cols = len(images)
  
  if cols > column_limit:
    rows = int(cols / column_limit) + 1
    cols = column_limit
    
  # Corner case check to ensure there's enough rows
  if (rows * cols) < len(images):
    rows += 1
  
  plt.figure()
  
  for entry, i in zip(images, range(len(images))):
    if "img" in entry:
      img = entry["img"]
      title = f"Image {i+1}" if "title" not in entry else entry["title"]
      
      plt.subplot(rows, cols, i+1)
      plt.imshow(img, cmap=cmap)
      plt.title(title)
      plt.xticks([]), plt.yticks([])
    
  plt.show(block=True)

def save_img(img, path="image.png", cmap="gray", fmt="png"):
  plt.imsave(fname=path, arr=img, cmap=cmap, format=fmt)

def main():
  logger.debug("Debug mode enabled.")
  bg_path = args.background
  fg_path = args.foreground
  use_sobel_x = args.use_sobel_x
  use_sobel_y = args.use_sobel_y

  if not os.path.exists(bg_path):
    raise OSError(f"{bg_path} does not exist. Path to an existing image file required.")

  if not os.path.exists(fg_path):
    raise OSError(f"{fg_path} does not exist. Path to an existing image file required.")

  logger.debug(f"Background image path: {bg_path}")
  logger.debug(f"Foreground image path: {fg_path}")

  # NOTE: Read as grayscale with cv2.IMREAD_GRAYSCALE as 2nd arg
  bg_img = cv2.imread(bg_path)
  fg_img = cv2.imread(fg_path)

  logger.debug(f"Background image shape: {bg_img.shape}")
  logger.debug(f"Foreground image shape: {fg_img.shape}")

  # Convert foreground image to grayscale
  gray_fg_img = cv2.cvtColor(fg_img, cv2.COLOR_BGR2GRAY)

  # Apply Sorbel filter to foreground image
  ddepth = cv2.CV_64F          # Output image depth
  dx = 1 if use_sobel_x else 0 # Order of derivative x
  dy = 1 if use_sobel_y else 0 # Order of derivative y
  ks = 7                       # Kernel size

  logger.debug("Sobel configuration:")
  logger.debug(f"|-- ddepth = {ddepth} (cv2.CV_64F)")
  logger.debug(f"|-- dx     = {dx}")
  logger.debug(f"|-- dy     = {dy}")
  logger.debug(f"|-- ks     = {ks}")
  extruded_fg_img = cv2.Sobel(gray_fg_img, ddepth, dx, dy, ksize=ks)

  # Resize foreground image to match the size of the background image
  # TODO Add ability (by user preference) to add padding around foreground image when image 
  #      would be stretched to fit the size of the background image (e.g., when aspect ratios don't match)
  original_fg_shape = fg_img.shape
  original_extruded_fg_shape = extruded_fg_img.shape
  h, w, _ = bg_img.shape # Extract height (rows), width (cols), channels (if colored)
  fg_img = cv2.resize(fg_img, (w, h), interpolation=cv2.INTER_AREA)
  extruded_fg_img = cv2.resize(extruded_fg_img, (w, h), interpolation=cv2.INTER_AREA)
  logger.info(f"Background image size is {bg_img.shape}")
  logger.info(f"Foreground image resized from {original_fg_shape} to {fg_img.shape}")
  logger.info(f"Extruded foreground image resized from {original_extruded_fg_shape} to {extruded_fg_img.shape}")
  logger.debug(f"Foreground image shape:          {fg_img.shape}")
  logger.debug(f"Background image shape:          {bg_img.shape}")
  logger.debug(f"Extruded foreground image shape: {extruded_fg_img.shape}")

  if DEBUG_MODE:
    display_all([
      { "img": fg_img, "title": "Foreground"},
      { "img": extruded_fg_img, "title": "Extruded Foreground" },
      { "img": bg_img, "title": "Background" },
    ], column_limit=3)

  # Save extruded image to disc so it can be read in with alpha channels and 
  # original grayscale look in tact. See https://stackoverflow.com/a/63091765
  # 
  # NOTE: Inverted grayscale is gray_r, which make extruded areas concave 
  #       rather than the desired convex appearance
  save_img(extruded_fg_img, 'extruded_fg_img.png')

  # Read the extruded image back in from disc using opencv to prepare for color transferring
  extruded_fg_img = cv2.imread('extruded_fg_img.png')

  if extruded_fg_img.shape == bg_img.shape:
    logger.debug("Extruded foreground image shape matches background image shape.")
  else:
    logger.debug("Extruded foreground image shape doesn't match background image shape.")
    
  logger.debug(f"|-- Foreground shape: {extruded_fg_img.shape}")
  logger.debug(f"|-- Background shape: {bg_img.shape}")

  if DEBUG_MODE:
    # Display new extruded image with color alpha channels
    # Disabling color map (cmap) for proof of concept that the new extruded image is colored
    display(
      extruded_fg_img, 
      "Resized Extruded Foreground Image with Channels",
      cmap=None
    )

  # **Goal:** Apply background image as a skin to the extruded foreground image
  # Use image blending to achieve desired result
  fg_weight = 0.55
  bg_weight = 0.65
  scalar = 20.0 # Scalar added to each sum when blending images
  blended_img = cv2.addWeighted(extruded_fg_img, fg_weight, bg_img, bg_weight, scalar)

  logger.info("Created final blended image.")

  if DEBUG_MODE:
    display(blended_img, title="Final Blended Image", cmap=None)

    # Display all images for comparison of start to finish
    display_all([
      { "img": extruded_fg_img, "title": "Foreground" },
      { "img": bg_img, "title": "Background" },
      { "img": blended_img, "title": "Blended" },
    ], column_limit=3)

  # Save final result to file
  # Create directory if it doesn't exit
  dir_name = os.path.dirname(args.outpath)
  path_is_dir_only = os.path.splitext(args.outpath)[1] == ''
  dir_exists = os.path.exists(dir_name)
  if not dir_exists and dir_name != '':
    os.makedirs(dir_name)

  outpath = args.outpath

  # Determine whether a file was specified in the outpath arg
  if path_is_dir_only:
    outpath = os.path.join(outpath, 'blended.png')

  save_img(blended_img, outpath)
  logger.info(f"See {outpath}")

if __name__ == "__main__":
  main()

