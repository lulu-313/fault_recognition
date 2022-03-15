from model import *
from tools import *
from sklearn.model_selection import train_test_split
from keras.models import *
import tensorflow._api.v2.compat.v1 as tf

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

path_home= os.path.abspath(os.path.join(os.getcwd(), "../../"))

trainSampleSrc=os.path.join(path_home,"data","ImageData","train","sample")
trainTargetSrc=os.path.join(path_home,"data","ImageData","train","target")


predictSampleSrc=os.path.join(path_home,"data","ImageData","predict","sample")
predictTargetSrc=os.path.join(path_home,"data","ImageData","predict","target")
epochsArray=[5,10]


trainSampleData=getData1(trainSampleSrc)/255
trainTargetData=getData1(trainTargetSrc)/255
predictSampleData=getData1(predictSampleSrc)/255
predictTargetData=getData1(predictTargetSrc)/255

trainX,testX,trainY,testY=train_test_split(trainSampleData,trainTargetData,random_state=42)
trainX=trainX.reshape(-1, 128, 128, 1)
testX=testX.reshape(-1, 128, 128, 3)

trainY=trainY.reshape((-1,128,128,1))
testY=testY.reshape((-1,128,128,1))
trainY1=numpy.where(trainY==0,1,0).reshape((-1,128,128,1))
testY1=numpy.where(testY==0,1,0).reshape((-1,128,128,1))

trainY=numpy.concatenate((trainY,trainY1),axis=3)
testY=numpy.concatenate((testY,testY1),axis=3)

print(trainX.shape,trainY.shape)

def train():
    rootDirSrc=os.path.join(path_home,"result_data","result_res_aspp1")
    for i in epochsArray:
        epochs = i
        resultDirSrc = os.path.join(rootDirSrc, "result_epochs_" + str(epochs))
        if not os.path.exists(resultDirSrc):
            os.makedirs(resultDirSrc)
        modelAndOthersArc = os.path.join(resultDirSrc, "modelAndOthersSrc")
        if not os.path.exists(modelAndOthersArc):
            os.makedirs(modelAndOthersArc)
        saveSrcFile = [
            os.path.join(modelAndOthersArc, "model.h5"),
            os.path.join(modelAndOthersArc, "weight.h5"),
            os.path.join(modelAndOthersArc, "a.png"),
            os.path.join(modelAndOthersArc, "a.txt")
        ]

        model = GFFResUNet()
        history = model.fit(trainX, trainY, epochs=epochs, batch_size=5, validation_data=(testX, testY))
        saveModelAndOthers(model, history, epochs, saveSrcFile)

        # model1 = load_model(saveSrcFile[0],{
        #     "binary_focal_loss_fixed":binary_focal_loss
        # })
        model.summary()

        print("加载模型成功")
        for i in range(len(os.listdir(predictSampleSrc))):
            predictImageArray = numpy.array(predictSampleData[i]).reshape(
                    (1, 128,128, 3))
            resultImage = numpy.array(model.predict(predictImageArray)).reshape(
                    (128,128, 2)) * 255.0
            resultImage0 = resultImage[:, :, 0]
            resultImage1 = resultImage[:, :, 1]
            print(resultImage0.shape, resultImage1.shape)
            resultImage = numpy.where(resultImage0 > resultImage1, 0, 255)
            print(resultImage0)
            print(resultImage1)
            resultImage = Image.fromarray(resultImage).convert("L")
            resultImage.save(os.path.join(resultDirSrc, os.listdir(predictSampleSrc)[i]))

def partly_train(pretrainedWeights):
    rootDirSrc = os.path.join(path_home, "result_data")
    for i in epochsArray:
        epochs = i
        resultDirSrc = os.path.join(rootDirSrc, "result_gff")
        if not os.path.exists(resultDirSrc):
            os.makedirs(resultDirSrc)
        modelAndOthersArc = os.path.join(resultDirSrc, "modelAndOthersSrc")
        if not os.path.exists(modelAndOthersArc):
            os.makedirs(modelAndOthersArc)
        saveSrcFile = [
            os.path.join(modelAndOthersArc, "model.h5"),
            os.path.join(modelAndOthersArc, "weight.h5"),
            os.path.join(modelAndOthersArc, "a.png"),
            os.path.join(modelAndOthersArc, "a.txt")
        ]

        model = GFFResUNet(pretrainedWeights)
        history = model.fit(trainX, trainY, epochs=epochs, batch_size=5, validation_data=(testX, testY))
        saveModelAndOthers(model, history, epochs, saveSrcFile)

        model.summary()

        print("加载模型成功")
        for i in range(len(os.listdir(predictSampleSrc))):
            predictImageArray = numpy.array(predictSampleData[i]).reshape(
                (1, 128, 128, 1))
            resultImage = numpy.array(model.predict(predictImageArray)).reshape(
                (128, 128, 2)) * 255.0
            resultImage0 = resultImage[:, :, 0]
            resultImage1 = resultImage[:, :, 1]
            print(resultImage0.shape, resultImage1.shape)
            resultImage = numpy.where(resultImage0 > resultImage1, 0, 255)
            print(resultImage0)
            print(resultImage1)
            resultImage = Image.fromarray(resultImage).convert("L")
            resultImage.save(os.path.join(resultDirSrc, os.listdir(predictSampleSrc)[i]))


if __name__ == '__main__':
    train()

