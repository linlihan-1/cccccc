import os
import shutil
OriPath = r'D:\CrossTransformers-PyTorch-main\miniImageNet\images'

for root, dirs, files in os.walk(OriPath, topdown=False):
    for name in files:
        print(os.path.join(root, name))
        filepath=os.path.join(root, name) #文件存放路径
        path1="D:\CrossTransformers-PyTorch-main\miniImageNet\ii"
        movepath=os.path.join(path1,name) #movepath：指定移动文件夹
        shutil.copyfile(filepath,movepath)