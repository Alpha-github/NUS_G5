import os
import shutil

src = ""  # Path of Dataset containing multiple labels of fruits and vegetables
dest = "" # Path of cleaned Dataset containing only Fresh and Rotten labels.

src_class = [i  for i in os.listdir(src) if i not in ['train','test']]
dest_class = [i  for i in os.listdir(dest) if i not in ['train','test']]

print(src_class, dest_class)

count_f = count_r = 1

for j in src_class:
    if j[0] == "F":
        spath = os.path.join(src,j)
        dpath = os.path.join(dest,"Fresh")
        for file in os.listdir(spath):
            sf = os.path.join(spath,file)
            df = os.path.join(dpath,str(count_f)+'.jpg')
            shutil.copy(sf,df)
            count_f+=1
    else:
        spath = os.path.join(src,j)
        dpath = os.path.join(dest,"Rotten")
        for file in os.listdir(spath):
            sf = os.path.join(spath,file)
            df = os.path.join(dpath,str(count_r)+'.jpg')
            shutil.copy(sf,df)
            count_r+=1

print(count_r,count_f)