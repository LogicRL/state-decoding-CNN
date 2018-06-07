import numpy as np
import cv2
from matplotlib import pyplot as plt

test_dir = './test_images/'
img1 = cv2.imread(test_dir + 'pumpkin.png',0)          # queryImage
img2 = cv2.imread(test_dir + 'raccoon_pumpkin.jpeg',0) # trainImage

# Initiate SIFT detector
orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)
#img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,flags=2)

plt.imshow(img3),plt.show()


#---------------------------------------------------------------------------


import numpy as np
import cv2
from matplotlib import pyplot as plt

test_dir = './test_images/'
img1 = cv2.imread(test_dir + 'man.png', 0)          # queryImage
img2 = cv2.imread(test_dir + 'monte.png', 0) # trainImage
#img1 = cv2.imread(test_dir + 'pumpkin.png')          # queryImage
#img2 = cv2.imread(test_dir + 'raccoon_pumpkin.jpeg') # trainImage

# Initiate SIFT detector
#sift = cv2.xfeatures2d.SIFT_create()
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)

plt.imshow(img3),plt.show()


#---------------------------------------------------------------------------



import numpy as np
import cv2
from matplotlib import pyplot as plt

test_dir = './test_images/'
img1 = cv2.imread(test_dir + 'ghost.png',0)          # queryImage
img2 = cv2.imread(test_dir + 'monte.png',0) # trainImage

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params,search_params)

matches = flann.knnMatch(des1,des2,k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = 0)

img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)

plt.imshow(img3,),plt.show()


#---------------------------------------------------------------------------


import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
 
# Read the main image
test_dir = './test_images/'
img_rgb = cv2.imread(test_dir + 'raccoon_pumpkin.jpeg')
 
# Convert it to grayscale
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
 
# Read the template
template = cv2.imread(test_dir + 'pumpkin.png',0)
 
# Store width and heigth of template in w and h
w, h = template.shape[::-1]
 
# Perform match operations.
res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
 
# Specify a threshold
threshold = 0.8
 
# Store the coordinates of matched area in a numpy array
loc = np.where( res >= threshold) 
 
# Draw a rectangle around the matched region.
for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)
print(img_rgb)
 
# Show the final image with the matched area.
plt.imshow(img_rgb,),plt.show()

