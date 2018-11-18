import sys
import imageNetDownloader as imd

imDown = imd.Downloader()
ids = []
wnid = input("Enter WNID of desired Synset: ")
title = input("Enter desired folder title: ")
size = int(input("Enter desired number of images to generate: "))
exNum = int(input("How many subSynsets would you like to exclude: "))
exclude = [None] * exNum
print("You will now be prompted to input Synsets you want excluded.")
for i in range(len(exclude)):
    exclude[i] = input("Wnid to exclude: ")
ids = imDown.get_subsets(wnid, exclude)
imDown.merge_synsets(ids, size, title)
#imDown.download_url("http://farm1.static.flickr.com/160/404748410_95c84be084.jpg", "./Cactus")
