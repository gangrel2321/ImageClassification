import sys
import os
import shutil
import urllib.request as urlReq
import urllib.parse as urlParse
import random
from PIL import Image

class Downloader:

    def __init__(self):
        pass

    #gets the list of urls for each wnid
    def _getUrls(self, wnid):
        urlLoc = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=" + str(wnid)
        urlOb = urlReq.urlopen(urlLoc)
        urlList = urlOb.read().decode("utf-8")
        urlList = urlList.split("\n")
        urlList = list(map(lambda x: x.rstrip("\r"), urlList))
        return urlList

    def get_subsets(self, wnid, exclude = []):
        wnid.strip()
        if len(wnid) < 3:
            return []
        url = "http://image-net.org/api/text/wordnet.structure.hyponym?wnid=" + wnid
        urlOb = urlReq.urlopen(url)
        urlList = urlOb.read().decode("utf-8")
        urlList = urlList.split("\n")
        urlList = list(map(lambda x: x.rstrip("\r"), urlList))
        subSet = []
        for url in urlList:
            if len(url) > 0 and url[0] == "-" and not (url[1:] in exclude):
                subSet.append(url[1:])
        return subSet

    #constructs a list of urls for new merged synset and downlds them
    def merge_synsets(self, ids, size, title, loc = None):
        if loc:
            os.chdir(loc)
        if os.path.isdir(title):
            choice = input("A directory with the name {0:} already exists, would you like to override it? (y/n): ".format(title)).lower()
            if choice == 'y':
                shutil.rmtree(title)
                print("Directory overriden.")
            else:
                print("Canceling merge.")
                return []
        os.makedirs(title)
        os.chdir(title)
        loc = os.getcwd()
        print("Constructing new Synset...")
        newSyn = []
        for i in range(len(ids)):
            newSyn += self._getUrls(ids[i])

        #size of mega synset we're sampling from
        newSynSize = len(newSyn)
        if size > newSynSize:
            error = "Error: Desired size exceeds that of combined Synset size." \
                    + "\nSetting desired size to combined Synset size."
            print(error)
            size = newSynSize
        #indices = random.sample(range(1,newSynSize),size)
        random.shuffle(newSyn)
        print("Construction Complete.")
        shuffledSyn = newSyn
        i = 0
        indices = []
        print("Downloading Images...")
        while i < size:
            url = shuffledSyn[i]

            try:
                indices.append(i)
                self.download_url(url, loc)
            except Exception:
                #that link didn't work so we need to increase the size
                indices.pop()
                size += 1
            i += 1
            if size == len(shuffledSyn):
                #we can't reach desired size due to too many image corruptions
                i = size
            print("Percent Complete: {0:.1f}".format(i/size * 100), end="\r")
        #might replace with operator itemgetter if speed is an issue
        finalSynUrls = list(map(shuffledSyn.__getitem__, indices))
        print("Image Download Complete.")
        return finalSynUrls


    def download_url(self, url, weg = None):

        urlOb = urlReq.urlopen(url)
        path = urlParse.urlsplit(url)[2]
        fileName = path.split("/")[-1] #os.path.basename(path)

        if not fileName:
            raise Exception("Could not locate file to download")

        if weg:
            fileName = os.path.join(weg, fileName)

        with open(fileName, 'wb') as file:
            file_size_dl = 0
            block_sz = 8192
            while True:
                buffer = urlOb.read(block_sz)
                if not buffer:
                    break
                file_size_dl += len(buffer)
                file.write(buffer)

        fileSize = os.path.getsize(fileName)
        fileType = fileName.split(".")[-1]

        if fileSize < 5000 or (fileSize < 10242 and fileSize > 10235):
            os.remove(fileName)
            raise Exception("Downloaded file is corrupt")

        elif not (fileType == "jpg" or fileType == "png"):
            os.remove(fileName)
            raise Exception("Downloaded file was not an image")

        if not self.resize_image(fileName):
            os.remove(fileName)
            raise Exception("Invalid Image")

        return fileName


    def resize_image(self, fileName):
        image_path = fileName
        image = Image.open(fileName).resize((224,224))
        image.save(image_path)
        if image.size != (224,224):
            return False
        return True
