import numpy
from PIL import Image
im = Image.open("IMG_0855.JPG")
im.show()
gray=numpy.array(im)
a = np.zeros([width,height,3],dtype=np.uint8)
for x in range (0,width):
	for y in range (0,height):
		a[x][y]=gray[x+xC][y+yC]
im=Image.fromarray(a)

im.show()