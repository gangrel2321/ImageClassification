import os

count = 0
for file in os.listdir(os.getcwd()):
    if os.path.splitext(file)[1] != ".py":
        os.rename(file, "n13108841_" + str(count) + ".png")
        count += 1
