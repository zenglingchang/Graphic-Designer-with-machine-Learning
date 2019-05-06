import numpy as np
from PIL import Image


def GetDesginImg(ElementList, DesignList):
    BackGroud = ElementList[0]
    (Width, Height) = BackGroud.size
    for i in range(1,len(ElementList)):
        (X0,Y0,ScaleX,ScaleY) = DesignList[i]
        TempImg = ElementList[i].resize((int(ScaleX*Width), int(ScaleY*Height)), Image.ANTIALIAS)
        r,g,b,a = TempImg.split()
        position = (int(X0*Width), int(Y0*Height))
        BackGroud.paste(TempImg, position + (position[0]+TempImg.size[0], position[1]+TempImg.size[1]), mask = a)
    return BackGroud
    