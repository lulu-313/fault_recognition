import os
import numpy
from PIL import Image
import matplotlib.pyplot as plt

dataDirSrc=r"D:\liyanchun\dizhenliefeng\data\imageData_n\train\sample"


def getData1(dataDirSrc):
    dataNameArray=os.listdir(dataDirSrc)
    dataSrcArray=[os.path.join(dataDirSrc,i) for i in dataNameArray]
    imageData=[]
    for i in range(len(dataNameArray)):
        imageData.append(numpy.array(Image.open(dataSrcArray[i]).convert("L")))
    return numpy.array(imageData)

def getData2(dataDirSrc):
    dataNameArray=os.listdir(dataDirSrc)
    dataSrcArray=[os.path.join(dataDirSrc,i) for i in dataNameArray]
    imageData=[]
    for i in range(len(dataNameArray)):
        imageData.append(numpy.array(Image.open(dataSrcArray[i]).convert("RGB")))
    return numpy.array(imageData)

def saveModelAndOthers(model,history,epochs,saveSrcFile):
    model.save(saveSrcFile[0])
    model.save_weights(saveSrcFile[1])
    epoch_array=list(range(epochs))
    acc = history.history["accuracy"]
    val_acc = history.history['val_accuracy']
    loss = history.history["loss"]
    val_loss = history.history['val_loss']
    plt.figure()
    plt.plot(epoch_array, acc, label="train_accuracy", color='r')
    plt.plot(epoch_array, val_acc, label="val_accuracy", color='b')
    plt.plot(epoch_array, loss, label="train_loss", color='r')
    plt.plot(epoch_array, val_loss, label="val_loss", color='b')
    plt.xlabel('epochs')
    plt.ylabel('y label')
    plt.title("loss and acc")
    plt.legend()
    plt.savefig(saveSrcFile[2])
    plt.close()
    f = open(saveSrcFile[3],"a")
    f.write(str(acc) + "\n")
    f.write(str(val_acc) + "\n")
    f.write(str(loss) + "\n")
    f.write(str(val_loss) + "\n")
    f.close()
