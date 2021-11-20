import cv2
import numpy as np
import random
import sys


def img_to_binary(img):
    '''
    converts image to binary image
    '''
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh = 127
    im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    return im_bw
    
def get_noise(img):
    '''
        creates noise image contains only 0 and 255 values
    '''
    noise = np.zeros(img.shape, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            noise[i][j] = random.randint(0,255)
    return noise

def do_criptography(img, noise):
    '''
        do asimmetric criptograpy with xor operator
    '''
    img_cript = np.bitwise_xor(img, noise)
    return img_cript

def save_image(img, path):
    cv2.imwrite(path, img)

def stackImages(imgArray,scale,lables=[]):
    '''
        stacks images in a single image
        from https://github.com/murtazahassan/OpenCV-Python-Tutorials-and-Projects/blob/master/Basics/Joining_Multiple_Images_To_Display.py
    '''
    sizeW= imgArray[0][0].shape[1]
    sizeH = imgArray[0][0].shape[0]
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (sizeW, sizeH), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (sizeW, sizeH), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d][c])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d][c],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

def main():

    imgpath = sys.path[0]+'/images/napoleon.png'
    img = cv2.imread(imgpath)
    # get filename from path
    filename = imgpath.split('/')[-1].split('.')[0]

    img_bin = img_to_binary(img)
    noise= get_noise(img_bin)
    img_cript = do_criptography(img_bin, noise)
    img_decript = do_criptography(img_cript, noise)
    
    # show image as stack and write with labels
    font = cv2.FONT_HERSHEY_PLAIN
    img_binTest = cv2.resize(img_bin, (0, 0), None, 0.5, 0.5)
    cv2.putText(img_binTest, 'Binary img', (0,20), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    noiseTest = cv2.resize(noise, (0, 0), None, 0.5, 0.5)
    cv2.putText(noiseTest, 'random noise', (0,20), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    img_criptTest = cv2.resize(img_cript, (0, 0), None, 0.5, 0.5)
    cv2.putText(img_criptTest, 'Encripted img', (0,20), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    img_decriptTest = cv2.resize(img_decript, (0, 0), None, 0.5, 0.5)
    cv2.putText(img_decriptTest, 'Decripted with random noise', (0,20), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    img_stackTest = stackImages([[img_binTest, noiseTest], [img_criptTest, img_decriptTest]], 1)
    cv2.imshow('Processed images', img_stackTest)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # show images
    # cv2.imshow('img', img)
    # cv2.imshow('img_bin', img_bin)
    # cv2.imshow('noise', noise)
    # cv2.imshow('cript', img_cript)
    # cv2.imshow('decript', img_decript)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # save images
    cv2.imwrite(sys.path[0]+'/out/'+filename+'_bin.png', img_bin)
    cv2.imwrite(sys.path[0]+'/out/'+filename+'_noise.png', noise)
    cv2.imwrite(sys.path[0]+'/out/'+filename+'_encripted.png', img_cript)
    cv2.imwrite(sys.path[0]+'/out/'+filename+'_decript.png', img_decript)

if __name__ == '__main__':
    main()