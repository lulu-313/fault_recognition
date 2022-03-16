
# fault_recognition
order to solve the problem of poor continuity and difficulty in fault identification in tectonically complex areas for oil and gas reservoir exploration and development, a seismic fault identification model with global feature fusion is proposed. 

Requirements:
python3.6
tensorflow2.2.0
keras2.3.4
numpy1.19.5

”data“ training and test data folders
”code“ Model training and testing source code folder
”result_data“ model and test result folders

"data/ImageData" is a 2D seismic profile
"data/train" is the raw synthetic seismic data
"code/data_generator" synthetic seismic data generation
"code/image_generator" 2D seismic profile generation
"code/model" gff-res-unet network model structure
"code/tool" data read model weight storage and other related tool functions
"code/trainmodel" is a program to train a gff-res-unet network model with training data
"code/trainmodel" uses the trained gff-res-unet network model to test the fault recognition effect

"result-data/result_gff/modelAndOtherSrc/a.png" accuracy and loss curve of gff-res-unet network model training
"result-data/result_gff/modelAndOtherSrc/a.txt" The accuracy and loss changes of gff-res-unet network model training
"result-data/result_gff/modelAndOtherSrc/model.h5" gff-res-unet network model structure
"result-data/result_gff/modelAndOtherSrc/weight.h5" gff-res-unet network model weights
