import sys
import urllib.request as urlReq
import urllib.parse as urlParse
from collections import defaultdict
from nltk.corpus import wordnet as wn

def getNeighbors(id):
    if len(id) < 3:
        return []
    url = "http://image-net.org/api/text/wordnet.structure.hyponym?wnid=" + id
    urlOb = urlReq.urlopen(url)
    urlList = urlOb.read().decode("utf-8")
    urlList = urlList.split("\n")
    urlList = list(map(lambda x: x.rstrip("\r"), urlList))
    neighbors = []
    for url in urlList:
        if len(url) > 0 and url[0] == "-":
            neighbors.append(url[1:])
    return neighbors

#top level id
wnid = "n00017222"
visited = defaultdict(lambda: False)
queue = []
queue.append(wnid)
visited[wnid] = True
#wn.of2ss('00017222n')
#.definition()
#

with open("imageNetHierachy.ttl", "w") as f:
    f.write("@prefix : <http://tetherless.rpi.edu/INO#> .\n@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .\n@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .\n@prefix owl: <http://www.w3.org/2002/07/owl#> .\n@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .\n")
    f.write(":" + wnid + " rdf:type" + " owl:Class .\n" )
    #breadth first search through imagenet structure
    while queue:
        wnid = queue.pop(0)
        print("Wnid:",wnid + " ")
        neighbors = getNeighbors(wnid)
        for i in neighbors:
            if visited[i] == False:
                queue.append(i)
                visited[i] = True
                f.write(":" + i + " rdf:type" + " owl:Class ;\n")
                f.write("    rdfs:subClassOf :" + wnid +" .\n")
