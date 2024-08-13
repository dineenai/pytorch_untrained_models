import infantTransforms.TinyEyesTransform as tinyeyes
import numpy as np
import os, random
from PIL import Image
from torchvision import transforms

# n02123394
# n02504013
in_imgs = "/data/ILSVRC2012/val_in_folders/n02504013"
img_list = os.listdir(in_imgs)
# remove ILSVRC2012_val_00010218.JPEG from list
# img_list.remove("ILSVRC2012_val_00010218.JPEG")

# create new folder called test_tiny_cats

if not os.path.exists("test_tiny_cats_d-30_50"):
    os.makedirs("test_tiny_cats_d-30_50")


# set epoch as list ranging from 0 to epochs 
epochs = np.arange(0, 60, 5)

for epoch in epochs:
    ages = ['week0', 'week4', 'week8', 'week12','week12','week24', 'week24','adult']

    # set number of epochs per age

    if epoch // 5 >= len(ages):
        age = ages[-1]
    else:
        age = ages[epoch // 5]


    
    print(f"Epoch: {epoch}, Age: {age}")
    

    for image in range(5):
        # # fpr week0 and week4
        # if epoch // 5 < 2:
        #     min_w, max_w = 30, 50

        # elif epoch // 5 >= 2:
        #     min_w, max_w = 10, 150
        
        min_w, max_w = 30, 50


        # randomly generate an int between 30 and 50
        width = np.random.randint(min_w, max_w)

        # # randomly generate float with one decimal place between 30 and 50
        # width = round(np.random.uniform(30, 50), 1)

        # height 2 * width * arctan (pi/3)
        dist = round(2 * width * np.arctan(np.pi/3), 2)
        print(f"Width: {width}, Dist: {dist}")



        # randomly choose stimulus image to apply the transfor to
        file = random.choice(img_list) 

      
        print(file)
       


        # Load stimulus image as PIL Image
        stim_img = Image.open(f'{in_imgs}/{file}')
         # get shape of image
        print(f'Size of stimulus image: {stim_img.size}')

        data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(224), #360
            tinyeyes.TinyEyes(age, width=width, dist=dist),
            # transforms.GaussianBlur(args.kernel, sigma=(args.gauss, args.gauss)),
            transforms.RandomHorizontalFlip(),
            # normalize #Rrmo
        ])

        # save transformed image to test_tiny_cats folder
        tiny_img = data_transforms(stim_img)
        # outdir = f'test_tiny_cats/{age}'
        # if not os.path.exists(outdir):
        #     os.makedirs(outdir)
        # tiny_img.save(f'{outdir}/{age}{file}')
        tiny_img.save(f'test_tiny_cats_d-30_50/{age}{file}')
        


     # randomly load any image from any subdirectory in 
     # 
     # randomly load any image from any subdirectory in /data/ILSVRC2012/val/n02123394
     # list files in directory
        # randomly select file
    # usse glob to get all files in directory
    # randomly select file



    # pick random folder from train



        

    # load random image from folder
    # apply TinyEyes transform
    # save image to folder
    # repeat for each epoch



# QUESTION
# How does this interact with LR