import os
from wand.image import Image
import shutter_stock
import adobe
import rf123
from wand.display import display

path = "/home/gollyzoom/Documents/CS229/instagram/"

dirList=os.listdir(path)

def convert():
    i = 0
    for subdir, dirs, files in os.walk(r'/home/gollyzoom/Documents/CS229/instagram/'):
        for filename in files:
            if filename.endswith(".jpg"):
                if(i % 1000 == 0):
                    print(i)
                i += 1
                #fname = subdir + os.sep + filename

                #print(filename)
                name = filename
                filepath = subdir + os.sep + filename
                with Image(filename=filepath) as image:
                    #print(name)
                    shutter_stock.apply_shutter_stock(image,'input_data/shutter_'+name)
                    adobe.apply_adobe(image,'input_data/adobe_'+name)
                    rf123.apply_123rf(image,'input_data/123rf_'+name)
                    if(image.size[0] < image.size[1]):
                        image.crop(0,0,image.size[0],image.size[0])
                    else:
                        image.crop(0,0,image.size[1],image.size[1])
                    image.resize(256,256)
                    image.save(filename ='output_data/shutter_'+name)
                    image.save(filename ='output_data/adobe_'+name)
                    image.save(filename ='output_data/123rf_'+name)

convert()
