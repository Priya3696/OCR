BiggestContour(contours):
using all our contours going to filter through the area so that we don't get too much noise. then going to check is it rectangle if its rectangle going further for find the biggest area and replace it when the biggest area found.

parameters:
contours - an outline representing or bounding the shape or form of something.

Returns:
find the rectangle shaped biggest contour and 4 marked edges

reorder(myPoints):
it will sorted out all points which we found use contouring.
sorting means arranging the points which is first or second.

Parameters:
myPoints - rectangle's edge points

Returns:
sorted points

stackImages(imgArray, scale, lables=None):
custom function to stack all images and it has option to add labels and add the name as well.

Parameters:
imgArray - all types of arrage images such as grey, thresholding...
scale - scale of the images

Returns:
stacked image
