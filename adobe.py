# Import library from Image
from wand.image import Image

# this function applies the Adobe watermark to an image specified by filename.
def apply_adobe(image,fname):
	with Image(filename ='watermarks/adobe.png') as adobe_watermark:
		crop_width = 0
		crop_height = 0

		if(image.size[0] < image.size[1]):
			image.crop(0,0,image.size[0],image.size[0])
		else:
			image.crop(0,0,image.size[1],image.size[1])
		image.resize(256,256)
		adobe_watermark.resize(int(image.size[0]*.6),int(adobe_watermark.size[1]*(.6*image.size[0]/adobe_watermark.size[0])))
		# Clone the image in order to process
		with image.clone() as watermarked_image:
			# apply the adobe watermark
			watermarked_image.watermark(adobe_watermark,.5,int(.5*image.size[0] - .5*adobe_watermark.size[0]),int(.5*image.size[1] - .5*adobe_watermark.size[1]))

			# Save the watermarked image
			watermarked_image.save(filename=fname)
