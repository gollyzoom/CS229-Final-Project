# Import library from Image
from wand.image import Image
def apply_shutter_stock(image,fname):
	with Image(filename ='watermarks/shutterstock_square.png') as water:
		#print(water.size)
		#print(image.size)
		crop_width = 0
		crop_height = 0

		#turn on for shutterstock and 123rf
		if(image.size[0] < image.size[1]):
			image.crop(0,0,image.size[0],image.size[0])
		else:
			image.crop(0,0,image.size[1],image.size[1])
		image.resize(256,256)
		water.resize(image.size[0],int(water.size[1]*(image.size[0]/water.size[0])))
		image.crop(0,0,water.size[0],water.size[1])
		water.distort('barrel_inverse', (0.2, 0.0, 0.0, 1.0))
		#turn on for adobe
		#water.resize(int(image.size[0]*.6),int(water.size[1]*(.6*image.size[0]/water.size[0])))

		#turn on for shutterstock
		#water.resize(water.size[0]*3,water.size[1]*3)

		# Clone the image in order to process
		with image.clone() as watermark:

			#turn on for shutterstock
			#watermark.watermark(water,.4, int(image.size[0]*-.5),int(image.size[1]*-.5))

			#turn on for 123rf
			watermark.watermark(water,.5, 0,0)

			#turn on for adobe
			#watermark.watermark(water,.5,int(.5*image.size[0] - .5*water.size[0]),int(.5*image.size[1] - .5*water.size[1]))

				# Save the image
			watermark.save(filename = fname)
			#watermark.save(filename ='watermarked_images/watermark_shutterstock.jpg')

