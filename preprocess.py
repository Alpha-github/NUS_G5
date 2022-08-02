import os
import shutil

base = r""  # Path for Dataset
train_split = 90

classes = [i for i in os.listdir(base) if i not in ['train','test']]
numclass = len(classes)

print(classes,numclass)

train_dir = os.path.join(base,'train')
test_dir = os.path.join(base,'test')

if 'train' not in os.listdir(base):
    os.mkdir(train_dir)
else:
    if len(os.listdir(train_dir))!=0:
        for j in os.listdir(train_dir):
            shutil.rmtree(os.path.join(train_dir,j))

if 'test' not in os.listdir(base):
    os.mkdir(test_dir)
else:
    if len(os.listdir(test_dir))!=0:
        for j in os.listdir(test_dir):
            shutil.rmtree(os.path.join(test_dir,j))

for i in classes:
    train_dest = os.path.join(train_dir,i)
    test_dest = os.path.join(test_dir,i)
    os.mkdir(train_dest)
    os.mkdir(test_dest)

    cl = os.path.join(base,i)
    num_images = len(os.listdir(cl))

    train = os.listdir(cl)[:int((train_split/100)*num_images)]
    test = os.listdir(cl)[int((train_split/100)*num_images):]

    for j in train:
        shutil.copy(os.path.join(cl,j),os.path.join(train_dest,j))
    
    for j in test:
        shutil.copy(os.path.join(cl,j),os.path.join(test_dest,j))

print("Data Cleaned !")
