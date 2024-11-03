#PS0001_PROJECT FULL PROGRAMM CHECK


#documentations:
#use img[r,c,0]: access red component
#use img[r,c,1]: access green component
#use img[r,c,2]: access blue component
#selection mask: Set of pixels from the image that have currently been selected by the user
#data type of selection mask: 2-D array (Bounded by size of image)
#mask[r,c] equals to 0 if the pixel located at row r and column c has not been selected

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def change_brightness(image, value):
    if -225 <= value <= 225: #bug capture
        img = image.copy() 
        #make a copy as requested
        img = np.clip(img + value, 0, 255) 
        #change brightness of the image and check for boundary values 
    return img
  
def change_contrast(image, value):
    if -225 <= value <= 225: #bug capture
        img = image.copy()
        F = (259 * (value + 255)) / (255 * (259 - value))
        img = np.clip(F * (img - 128) + 128, 0, 255)
    return img

def grayscale(image):
    img = image.copy()
    gray_values = 0.3 * img[:, :, 0] + 0.59 * img[:, :, 1] + 0.11 * img[:, :, 2]
    img[:, :, 0] = img[:, :, 1] = img[:, :, 2] = gray_values 
    # Multiplies 0.3 with all first entry of the array (R value), and so on
    return img


def blur_effect(image):
    kernel = np.array([[0.0625, 0.125, 0.0625],
                       [0.125,  0.25,  0.125],
                       [0.0625, 0.125, 0.0625]])
    img = image.copy()
    r = len(img) #column of image
    c = len(img[0]) # rows of image

    if r <= 2 or c<=2: #if the image less than 3x3 then cannot convolve
        return img
    for colour in range(2):
        for i in range(1,r-2): #make 3x3 neighbour of pixel i,j, i in [2,c-2] j in [2,r-2]
            for j in range(1,c-2):
                p = img[i,j] #access the pixel i,j, then identify the 3x3 pixels
                M = image[i-1:i+2, j-1:j+2, colour] #3x3 region around the pixel (FOR EACH R,G,B)
                img[i,j] = np.clip(np.sum(kernel*M), 0, 255).astype(np.unit8)
    return img

def edge_detection(image):
    kernel = np.array([[-1, -1, -1],
                       [-1,  8,  -1],
                       [-1, -1, -1]])
    img = image.copy()
    r = len(img) #column of image
    c = len(img[0]) # rows of image

    if r <= 2 or c<=2: #if the image less than 3x3 then cannot convolve
        return img
    for colour in range(2):
        for i in range(1,r-2): #make 3x3 neighbour of pixel i,j, i in [2,c-2] j in [2,r-2]
            for j in range(1,c-2):
                p = img[i,j] #access the pixel i,j, then identify the 3x3 pixels
                M = image[i-1:i+2, j-1:j+2, colour] #3x3 region around the pixel (FOR EACH R,G,B)
                img[i,j] = np.clip(np.sum(kernel*M), 0, 255).astype(np.unit8)
    return img

def embossed(image):
    kernel = np.array([[-1,-1, 0],
                       [-1, 0, 1],
                       [ 0, 1, 1]])
    img = image.copy()
    r = len(img) #column of image
    c = len(img[0]) # rows of image

    if r <= 2 or c<=2: #if the image less than 3x3 then cannot convolve
        return img
    for colour in range(2):
        for i in range(1,r-2): #make 3x3 neighbour of pixel i,j, i in [2,c-2] j in [2,r-2]
            for j in range(1,c-2):
                p = img[i,j] #access the pixel i,j, then identify the 3x3 pixels
                M = image[i-1:i+2, j-1:j+2, colour] #3x3 region around the pixel (FOR EACH R,G,B)
                img[i,j] = np.clip(np.sum(kernel*M), 0, 255).astype(np.unit8)
    return img

    return np.array([]) # to be removed when filling this function

def rectangle_select(image, x, y):
    return np.array([]) # to be removed when filling this function

def magic_wand_select(image, x, thres):
    return np.array([])

def compute_edge(mask):           
    rsize, csize = len(mask), len(mask[0]) 
    edge = np.zeros((rsize,csize))
    if np.all((mask == 1)): return edge        
    for r in range(rsize):
        for c in range(csize):
            if mask[r][c]!=0:
                if r==0 or c==0 or r==len(mask)-1 or c==len(mask[0])-1:
                    edge[r][c]=1
                    continue
                
                is_edge = False                
                for var in [(-1,0),(0,-1),(0,1),(1,0)]:
                    r_temp = r+var[0]
                    c_temp = c+var[1]
                    if 0<=r_temp<rsize and 0<=c_temp<csize:
                        if mask[r_temp][c_temp] == 0:
                            is_edge = True
                            break
    
                if is_edge == True:
                    edge[r][c]=1
            
    return edge

def save_image(filename, image):
    img = image.astype(np.uint8)
    mpimg.imsave(filename,img)

def load_image(filename):
    img = mpimg.imread(filename)
    if len(img[0][0])==4: # if png file
        img = np.delete(img, 3, 2)
    if type(img[0][0][0])==np.float32:  # if stored as float in [0,..,1] instead of integers in [0,..,255]
        img = img*255
        img = img.astype(np.uint8)
    mask = np.ones((len(img),len(img[0]))) # create a mask full of "1" of the same size of the laoded image
    img = img.astype(np.int32)
    return img, mask

def display_image(image, mask):
    # if using Spyder, please go to "Tools -> Preferences -> IPython console -> Graphics -> Graphics Backend" and select "inline"
    tmp_img = image.copy()
    edge = compute_edge(mask)
    for r in range(len(image)):
        for c in range(len(image[0])):
            if edge[r][c] == 1:
                tmp_img[r][c][0]=255
                tmp_img[r][c][1]=0
                tmp_img[r][c][2]=0
 
    plt.imshow(tmp_img)
    plt.axis('off')
    plt.show()
    print("Image size is",str(len(image)),"x",str(len(image[0])))

def menu():
    
    img = mask = np.array([])  # No image loaded at the start
    
    while True:
        if img.size == 0:
            print("What do you want to do ?")
            print("e - exit")
            print("l - load a picture")
        else:
            print("What do you want to do ?")
            print("e - exit")
            print("l - load a picture")
            print("s - save the current picture")
            print("1 - adjust brightness")
            print("2 - adjust contrast")
            print("3 - apply grayscale")
            print("4 - apply blur")
            print("5 - edge detection")
            print("6 - embossed")
            print("7 - rectangle select")
            print("8 - magic wand select")
        
        choice = input("Your choice: ").strip()
        
        if choice == 'e':
            break
        elif choice == 'l':
            filename = input("Enter the filename to load: ").strip()
            try:
                img, mask = load_image(filename)
                display_image(img, mask)
            except:
                print("Error loading image.")
        elif choice == 's' and img.size != 0:
            filename = input("Enter the filename to save: ").strip()
            save_image(filename, img)
        elif choice == '1' and img.size != 0:
            try:
                value = int(input("Enter brightness value (-255 to 255): ").strip())
                img = change_brightness(img, value)
                display_image(img, mask)
            except:
                print("Invalid input.")
        elif choice == '2' and img.size != 0:
            try:
                value = int(input("Enter contrast value (-255 to 255): ").strip())
                img = change_contrast(img, value)
                display_image(img, mask)
            except:
                print("Invalid input.")
        elif choice == '3' and img.size != 0:
            img = grayscale(img)
            display_image(img, mask)
        elif choice == '4' and img.size != 0:
            img = blur_effect(img)
            display_image(img, mask)
        elif choice == '5' and img.size != 0:
            img = edge_detection(img)
            display_image(img, mask)
        elif choice == '6' and img.size != 0:
            img = embossed(img)
            display_image(img, mask)
        else:
            print("Invalid choice or no image loaded.")
  
       
if __name__ == "__main__":
    menu()