import os, io, sys, copy, base64, random
import numpy as np
from PIL import Image

PersonDict = {
    "cute":0,
    "terror":1,
    "fashion":2,
    "business":3,
    "festive":4
}

def Base642Array(base64Data, type = 'RGB'):
    imgdata = base64.b64decode(base64Data.replace('data:image/png;base64,',''))
    image = io.BytesIO(imgdata)
    img = Image.open(image).convert(type)
    img = img.resize((192,256), Image.ANTIALIAS)
    TempArray = np.array(img)
    return TempArray
    
def Base642Img(base64Data, type = 'RGB'):
    imgdata = base64.b64decode(base64Data.replace('data:image/png;base64,',''))
    image = io.BytesIO(imgdata)
    img = Image.open(image)
    img = img.convert(type)
    img = img.resize((192,256), Image.ANTIALIAS)
    return img
    
def Img2Array(Img, type = 'RGB'):
    Img = Img.convert(type)
    TempArray = np.array(Img)
    return TempArray
    
def GetDesginImg(ElementList, DesignList):
    BackGroud = copy.deepcopy(ElementList[0])
    (Width, Height) = BackGroud.size
    for i in range(1,len(ElementList)):
        (X0,Y0,ScaleX,ScaleY) = DesignList[i]
        TempImg = ElementList[i].resize((int(ScaleX*Width), int(ScaleY*Height)), Image.ANTIALIAS)
        r,g,b,a = TempImg.split()
        position = (int(X0*Width), int(Y0*Height))
        BackGroud.paste(TempImg, position + (position[0]+TempImg.size[0], position[1]+TempImg.size[1]), mask = a)
    return BackGroud.convert("RGB")
    
def LoadingTrainingData():
    ImgTrainingPath = os.path.join(sys.path[0], r'data/train')
    dict = {
        "Labels": [],
        "Imgs" : []
    }
    for Dir in os.listdir(ImgTrainingPath):
        Label = [ 1 if i == PersonDict[Dir] else 0 for i in range(0,5)]
        for Img in os.listdir(os.path.join(ImgTrainingPath, Dir)):
            ImgSrc = os.path.join(ImgTrainingPath , Dir, Img)
            dict['Imgs'].append(Img2Array(Image.open(ImgSrc)).reshape([256*192*3]))
            dict['Labels'].append(Label)
    dict['Imgs'] = np.array(dict['Imgs'])
    dict['Labels'] = np.array(dict['Labels'])
    print("Training DataSet Size:",len(dict['Imgs']))
    return dict
    
def LoadingTestingData():
    ImgTestingPath = os.path.join(sys.path[0], r'data/test')
    dict = {
        "Labels": [],
        "Imgs" : []
    }
    for Dir in os.listdir(ImgTestingPath):
        Label = [ 1 if i == PersonDict[Dir] else 0 for i in range(0,5)]
        for Img in os.listdir(os.path.join(ImgTestingPath, Dir)):
            ImgSrc = os.path.join(ImgTestingPath, Dir, Img)
            dict['Imgs'].append(Img2Array(Image.open(ImgSrc)).reshape([256*192*3]))
            dict['Labels'].append(Label)
    dict['Imgs'] = np.array(dict['Imgs'])
    dict['Labels'] = np.array(dict['Labels'])
    return dict
    
def ClearDataSet():
    ImgTestingPath = os.path.join(sys.path[0], r'data/test')
    ImgTrainingPath = os.path.join(sys.path[0], r'data/train')
    for Path in [ImgTestingPath, ImgTrainingPath]:
        for Dir in os.listdir(Path):
            for Img in os.listdir(os.path.join(Path,Dir)):
                os.remove(os.path.join(Path, Dir, Img))
    print('Clear success!')
    
if __name__ == '__main__':
    ImgRoot = os.path.join(sys.path[0], 'data')
    ImgBuffer = os.path.join(ImgRoot, 'src')
    ImgTestPath = os.path.join(ImgRoot, 'test')
    ImgTrainPath = os.path.join(ImgRoot, 'train')
    for Dir in os.listdir(ImgBuffer):
        os.mkdir(os.path.join(ImgTestPath,Dir))
        os.mkdir(os.path.join(ImgTrainPath,Dir))
        for Img in os.listdir(os.path.join(ImgBuffer,Dir)):
            ImgSrc = os.path.join(ImgBuffer, Dir, Img)
            ImgDst = os.path.join(ImgTrainPath if random.random() > 0.2 else ImgTestPath, Dir, Img)
            im = Image.open(ImgSrc)
            im = im.resize((192,256), Image.ANTIALIAS)
            im.convert("RGB")
            im.save(ImgDst)

            
        
    
    