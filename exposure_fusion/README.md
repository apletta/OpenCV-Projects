# Exposure Fusion

<img src="https://github.com/apletta/OpenCV-Projects/blob/master/exposure_fusion/exposure_fusion.png" alt="exposure fusion example" width="100%">

Notes:
1. Adding CLAHE could further improve feature detection - [A Generic Image Processing Pipeline for Enhancing Accuracy and Robustness of Visual Odometry](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9695527/)
2. In theory HDR likely has improved feature detection performance compared to single-frame auto exposure, even optimizing for gradient information as in [Auto-adjusting Camera Exposure for Outdoor Robotics
using Gradient Information](https://joonyoung-cv.github.io/assets/paper/14_iros_auto_adjusting.pdf), due to capturing more full scene features than a single exposure alone

## References
- "Exposure Fusion" (2007), Tom Mertens, Jan Kautz, Frank Van Reeth: https://web.stanford.edu/class/cs231m/project-1/exposure-fusion.pdf
> Original paper from Mertens for exposure fusion
- "High Dynamic Range (HDR)" (4.10.0-dev), OpenCV: https://docs.opencv.org/4.x/d2/df0/tutorial_py_hdr.html
> OpenCV example of using exposure fusion, including Mertens and other approaches
- "Depth from HDR imaging" (2024), Orbbec: https://www.orbbec.com/docs/g330-depth-from-hdr-imaging/
> Examples of HDR performance for stereo vision

## Images/St. Louis 
- Publicly sourced from Kevin McCoy on Wikipedia: https://commons.wikimedia.org/wiki/High_dynamic_range_images
