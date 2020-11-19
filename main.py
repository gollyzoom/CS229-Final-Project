# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import os
from wand.image import Image
import shutter_stock
import adobe
import rf123
from wand.display import display

path = "/home/gollyzoom/Documents/CS229/images/"

dirList=os.listdir(path)

def convert():
    for fname in dirList:
        print(fname)
        with Image(filename=path+fname) as image:
            shutter_stock.apply_shutter_stock(image,'X_train/shutter_'+fname)
            adobe.apply_adobe(image,'X_train/adobe_'+fname)
            rf123.apply_123rf(image,'X_train/123rf_'+fname)

            #print img.size

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    convert()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
