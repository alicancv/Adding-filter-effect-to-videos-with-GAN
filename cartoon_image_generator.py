import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


original_images_path = "D:/Bitirme Projesi1/Pix2Pix/bitirme1_final_documents/celeba_hq/train/female"
filtered_images_path = "D:/Bitirme Projesi1/Pix2Pix/bitirme1_final_documents/filter_dataset"

def  applyFilter(name, original_path, filtered_path, count=1000):
    a = 1
    for i in os.listdir(original_path):
        img_name = name + str(a) +".png";
        
        path_1=os.path.join(original_path, i)
        image = cv2.imread(path_1) 
        image_oil = cv2.xphoto.oilPainting(src=image, size=1, dynRatio=100)
        image_inferno = cv2.cvtColor(image_oil, cv2.COLOR_BGR2GRAY)
        plt.contourf(np.flipud(image_inferno), levels=3, cmap="inferno")
        plt.axis("off")
        plt.savefig(img_name, bbox_inches="tight", pad_inches=0)
        print(img_name)
        image_inferno = cv2.resize(cv2.imread(img_name), (256, 256)) 
        image = cv2.resize(image, (256, 256))
        output = np.concatenate((image, image_inferno), axis=1)
        cv2.imwrite(filtered_path + "/" + img_name[:len(img_name)-4] +"_concatenated.png", output)
        a += 1
    
        if(a > count):
            break;
            
if __name__ == "__main__":
    applyFilter("inferno_female_", original_images_path, filtered_images_path, 5)