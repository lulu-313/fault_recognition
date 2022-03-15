from model import *
from tools import *
from keras.models import *
import tensorflow._api.v2.compat.v1 as tf

gpu_options = tf.GPUOptions(allow_growth=True)

def testmodel(predictSampleData,rootDirSrc):
    resultDirSrc = os.path.join(rootDirSrc, "result_gff")
    modelAndOthersArc = os.path.join(resultDirSrc, "modelAndOthersSrc")
    saveSrcFile = [
        os.path.join(modelAndOthersArc, "model.h5"),
        os.path.join(modelAndOthersArc, "weight.h5"),
        os.path.join(modelAndOthersArc, "a.png"),
        os.path.join(modelAndOthersArc, "a.txt")
    ]
    model = load_model(saveSrcFile[0],{"binary_focal_loss_fixed":binary_focal_loss
    })
    model.summary()

    print("加载模型成功")
    for i in range(len(os.listdir(predictSampleSrc))):
        predictImageArray = numpy.array(predictSampleData[i]).reshape(
                (1, 128,128, 1))
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

if __name__ == '__main__':
    path_home = os.path.abspath(os.path.join(os.getcwd(), "../"))
    rootDirSrc = os.path.join(path_home, "result_data")

    predictSampleSrc = os.path.join(path_home, "data", "ImageData", "predict", "sample")
    predictTargetSrc = os.path.join(path_home, "data", "ImageData", "predict", "target")

    predictSampleData = getData1(predictSampleSrc) / 255
    testmodel(predictSampleData,rootDirSrc)















