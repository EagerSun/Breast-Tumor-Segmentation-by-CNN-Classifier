import os 
import sys
import cv2
import h5py
import numpy as np
import pandas as pd
import statistics
from PIL import Image, ImageFilter
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import load_model

class DataProcessing():
    def __init__(self, mode = 'unbiase', splitNum4Test = 28, threshold = 0.5):
        self.mode = mode
        self.splitNum4Test = splitNum4Test
        self.threshold = threshold
#------------------------------------------------------------------------------------------------------------------------------
#-------Deal with dataset from '/data' address---------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
   
    def addressList(self):
        '''
        Return the sub-address with ID of patients from data folder. Each ID file contains sub-file as 1 and 0 where 1 includes images that represent tumor and 0 includes images that represent normal tissue.
        All the images from a ID files could be reconstructed into a original slide image of patient's breast.
        '''
        mainAdd = os.path.join(os.getcwd(), 'data')
        aL = os.listdir(mainAdd)
        aL = [os.path.join(mainAdd, i) for i in aL if i != '.ipynb_checkpoints']
        return aL
    
    def classAddress(self):
        '''
        Return the sub-address of 1 and 0 files for all IDs of patients.
        '''
        fileAdd = self.addressList()
        classAdd = [[os.path.join(i, j) for j in os.listdir(i)] for i in fileAdd]
        return classAdd
    
    def imageAddress(self):
        '''
        Return the addresses of images for all IDs of patients.
        '''
        classAdd = self.classAddress()
        imageAddSplit = [[[os.path.join(j, k) for k in os.listdir(j)] for j in i] for i in classAdd]
        imageAddCombine = [i[0] + i[1] for i in imageAddSplit]
        return imageAddSplit, imageAddCombine
    
    def extractInformation(self, imageAddItem): 
        '''
        Return the location (x, y) of correspond image from imageAddItem for original slide image of patient and correspond label.
        In this case, label 1 indicates this image portion contains tumor, else this image portion contains normal tissue. 
        '''
        # This function was also used in dealing with extracting information from images in '/crop' address
        imageAddSplit1 = imageAddItem.split('\\')[-1]
        imageAddSplit2 = imageAddSplit1.split('.')[0]
        imageAddSplit3 = imageAddSplit2.split('_')[2:]
        x = int(imageAddSplit3[0][1:])
        y = int(imageAddSplit3[1][1:])
        label = float(int(imageAddSplit3[2][-1]))
        return x, y, label
    
    def maskPortion4imagePortion(self, label, shape):
        '''
        Return a mask for image based on its label:
        If image contain tumor, mask with similar shape would be estalished as black;
        Else mask with similar shape would be estalished as white;
        '''
        if label == 0:
            maskPortion = np.ones((shape[0], shape[1], shape[2]), dtype = np.float32) * 255.0
        else:
            maskPortion = np.zeros((shape[0], shape[1], shape[2]), dtype = np.float32) * 255.0
        return maskPortion
    
    def imageInformationSingle(self, imageAdd):
        '''
        Return the correspond whole image & mask for ID of patient.
        Both image & mask are in PIL.Imgae RGB format.
        '''
        xMax, yMax = 0, 0
        for i in imageAdd:
            x, y, label = self.extractInformation(i)
            if x > xMax:
                xMax = x
            if y > yMax:
                yMax = y
        imageHolder = np.ones((yMax+50, xMax+50, 3), dtype = np.float32)*255.0
        maskHolder = np.ones((yMax+50, xMax+50, 3), dtype = np.float32)*255.0
        for i in imageAdd:
            x, y, label = self.extractInformation(i)
            imagePortion = np.asarray(Image.open(i), dtype = np.float32)
            maskPortion = self.maskPortion4imagePortion(label = label, shape = imagePortion.shape)
            imageHolder[y:y+imagePortion.shape[0], x:x+imagePortion.shape[1], :] = imagePortion
            maskHolder[y:y+imagePortion.shape[0], x:x+imagePortion.shape[1], :] = maskPortion
        
        imageHolder = Image.fromarray(np.uint8(imageHolder), 'RGB')
        maskHolder = Image.fromarray(np.uint8(maskHolder), 'RGB')
        return imageHolder, maskHolder
    
    def sample4ImagePortion(self, imageHolder, maskHolder, iD, numPostive, cropAdd):
        '''
        Given the original slide image and correspond mask for iD,
        Specific number of image portions label as 0 were further need to maintain whole training data/label to be label-uniformed.
        Given numPostive, numPostive image portions label as 0 could be cropped from imageholder and store in address: cropAdd, respectively.
        In this case, the unbiased mode here intend to do over-sampling.
        '''
        mask = np.asarray(maskHolder, dtype = np.float32)/255.
        image = np.asarray(imageHolder, dtype = np.float32)
        maskSlice = mask[:, :, 0]
        
        l = []
        (y, x) = np.where(maskSlice == 0)
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        for i in range(x.shape[0]):
            xL = x[idx[i]]
            yL = y[idx[i]]
            if np.sum(maskSlice[yL:yL+50, xL:xL+50]) == 0:
                l.append([yL, xL])   
            if len(l) >= int(1.52 * numPostive):
                break
                
        for i in l:
            iDAdd = os.path.join(cropAdd, iD)
            if not os.path.exists(iDAdd):
                os.mkdir(iDAdd)
            storeAdd = os.path.join(iDAdd, iD + '_idx5_' + 'x' + str(i[1]) + '_' + 'y' + str(i[0]) + '_class1.png')
            imagePortion = image[i[0]:i[0]+50, i[1]:i[1]+50, :]
            Image.fromarray(np.uint8(imagePortion), 'RGB').save(storeAdd)
            
#------------------------------------------------------------------------------------------------------------------------------
#-----------End Here...-----------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
            
    def sample4unbiased(self):
        '''
        This function is the summary version of sample4ImagePortion().
        In this function, extra cropped labeled 1 image portion would be collected from slide image from each ID and store in cropAdd, respectively.
        cropAdd was defined in this function.
        the format storing images in cropAdd should be similar with data except 0 and 1 images were collected in ID instead for crop. 
        '''
        aL = self.addressList()
        iD = [i.split('\\')[-1] for i in aL]
        
        splitAdd, combineAdd = self.imageAddress()
        postiveNum = [len(i[1]) for i in splitAdd] 
        
        cropAdd = os.path.join(os.getcwd(), 'crop')
        if not os.path.exists(cropAdd):
            os.mkdir(cropAdd)
        else:
            print("Cropping have already been finished...\n")
            
        
        print("Cropping have already start...")
        for index, i in enumerate(combineAdd):
            imageHolder, maskHolder = self.imageInformationSingle(i)
            self.sample4ImagePortion(imageHolder, maskHolder, iD[index], postiveNum[index], cropAdd)
            if (index+1) % 10 == 0:
                sys.stdout.write("\r\tProcess have been finished {0:.2%}".format(float((index+1)/len(combineAdd))))
                sys.stdout.flush()
                
        print("\nCropping had finished.\n")
        
#------------------------------------------------------------------------------------------------------------------------------
#-----------Deal with dataset from '/data' and '/crop' addresses-----------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
    def addressListCrop(self):
        '''
        Return the sub-address with ID of patients from crop folder. 
        '''
        mainAdd = os.path.join(os.getcwd(), 'crop')
        aL = os.listdir(mainAdd)
        aL = [os.path.join(mainAdd, i) for i in aL if i != '.ipynb_checkpoints']
        return aL
    
    def imageAddressCrop(self):
        '''
        Return the cropped image addresses from all IDs of patient. 
        '''
        iDAdd = self.addressListCrop()
        imageAddCombine = [[os.path.join(i, j) for j in os.listdir(i)] for i in iDAdd]
        return imageAddCombine
    
    def prepareTrainData(self):
        '''
        In this function, image porions and correspond labels from different ID could be collected and concatenated in arrays.
        Besides, train/test split would be finished where train data/labels, test data/labels could be stored in splited files(.h5).
        Whta's more, if self.mode is unbiase, data could be collected from both data and crop folder, else data data could be collected only from data folder.
        '''
        #----------------------------------------------------------------------------------------------------------------------
        #Judge whether  .h5 file and .csv file are exist.----------------------------------------------------------------------
        #----------------------------------------------------------------------------------------------------------------------
        dataStoreAddDir = os.path.join(os.getcwd(), 'dataStore')
        if not os.path.exists(dataStoreAddDir):
            os.mkdir(dataStoreAddDir)
        trainDataAddDir = os.path.join(dataStoreAddDir, 'trainData')
        if not os.path.exists(trainDataAddDir):
            os.mkdir(trainDataAddDir)
        testDataAddDir = os.path.join(dataStoreAddDir, 'testData')
        if not os.path.exists(testDataAddDir):
            os.mkdir(testDataAddDir)
        
        if os.path.exists(os.path.join(trainDataAddDir, 'trainData.h5')) and os.path.exists(os.path.join(testDataAddDir, 'testiD.csv')):
            print("trainData.h5 and testiD.csv had been produced in correspond dictionary.")
            return
        
        
        #----------------------------------------------------------------------------------------------------------------------
        #End Here.----------------------------------------------------------------------
        #----------------------------------------------------------------------------------------------------------------------
        if self.mode == 'unbiase' and not os.path.exists(os.path.join(os.getcwd(), 'crop')):
            print("The mode is unbiase, cropping process is begin...\n")
            self.sample4unbiased()
            
        print("Train/Test splitting is start...")
        _, imageAddListData = self.imageAddress()
        imageAddListCrop = self.imageAddressCrop()
        iD = [i.split('\\')[-1] for i in self.addressList()]
        
        idx = np.arange(len(iD))
        np.random.shuffle(idx)
        testID =[iD[i] for i in list(idx[0:self.splitNum4Test])]
        
        trainImageAddPre = []
        testImageAddPre = []
        for index, i in enumerate(iD):
            if i not in testID:
                if self.mode == 'biase':
                    trainImageAddPre += imageAddListData[index]
                else:
                    trainImageAddPre += imageAddListData[index] + imageAddListCrop[index]
            else:
                if self.mode == 'biase':
                    testImageAddPre += imageAddListData[index]
                else:
                    #testImageAddPre += imageAddListData[index] + imageAddListCrop[index]
                    testImageAddPre += imageAddListData[index]
                
        trainImageAdd = []
        for i in trainImageAddPre:
            imageItem = np.asarray(Image.open(i), dtype = np.float32)
            if imageItem.shape[0] == 50 and imageItem.shape[1] == 50 and imageItem.shape[2] == 3:
                trainImageAdd.append(i)
            else:
                continue
               
        trainData = np.zeros((len(trainImageAdd), 50, 50, 3), dtype = np.float32)
        trainLabel = np.zeros((len(trainImageAdd), ), dtype = np.float32)
        for index, i in enumerate(trainImageAdd):
            imageItem = np.asarray(Image.open(i), dtype = np.float32)
            trainData[index, :, :, :] = np.asarray(Image.open(i), dtype = np.float32)
            trainLabel[index] = float(int(i.split('.')[-2][-1]))
        
        trainData = trainData/255.
        
        print("Train data storing is start...")
        hFile = h5py.File(os.path.join(trainDataAddDir, 'trainData.h5'), 'w')
        hFile.create_dataset('trainData', data=trainData)
        hFile.create_dataset('trainLabel', data=trainLabel)
        hFile.close()
        
        print("Train data storing is over")
        testDataFrame = pd.DataFrame()
        testDataFrame['iD'] = testID
        testDataFrame.to_csv(os.path.join(testDataAddDir, 'testiD.csv'))
        
        testImageAdd = []
        for i in testImageAddPre:
            imageItem = np.asarray(Image.open(i), dtype = np.float32)
            if imageItem.shape[0] == 50 and imageItem.shape[1] == 50 and imageItem.shape[2] == 3:
                testImageAdd.append(i)
            else:
                continue
               
        testData = np.zeros((len(testImageAdd), 50, 50, 3), dtype = np.float32)
        testLabel = np.zeros((len(testImageAdd), ), dtype = np.float32)
        for index, i in enumerate(testImageAdd):
            imageItem = np.asarray(Image.open(i), dtype = np.float32)
            testData[index, :, :, :] = np.asarray(Image.open(i), dtype = np.float32)
            testLabel[index] = float(int(i.split('.')[-2][-1]))
        
        testData = testData/255.
        
        print("Train data storing is start...")
        hFile2 = h5py.File(os.path.join(trainDataAddDir, 'testData.h5'), 'w')
        hFile2.create_dataset('testData', data=testData)
        hFile2.create_dataset('testLabel', data=testLabel)
        hFile2.close()
        print("Train/Test splitting is finish...")
        return 
    
    def readDataLabel(self, mode = 'train'):
        '''
        Return the training data/label or testing data/label depends on the mode defined in input.
        '''
        dataStoreAddDir = os.path.join(os.getcwd(), 'dataStore')
        trainDataAddDir = os.path.join(dataStoreAddDir, 'trainData')
        if mode == 'train':
            hfileAddress = os.path.join(trainDataAddDir, 'trainData.h5')
            hFile = h5py.File(hfileAddress, 'r')
            Data = hFile.get('trainData').value
            Label = hFile.get('trainLabel').value
        else:
            hfileAddress = os.path.join(trainDataAddDir, 'testData.h5')
            hFile = h5py.File(hfileAddress, 'r')
            Data = hFile.get('testData').value
            Label = hFile.get('testLabel').value
        return Data, Label
        
    def prepareTestData(self, index):
        '''
        Return the training data/label or testing data/label depends on the mode defined in input.
        '''
        
        dfTest = pd.read_csv(os.path.join(os.getcwd(), 'dataStore', 'testData', 'testiD.csv'))
        testiD = dfTest['iD'].tolist()
        
        targetID = testiD[index]
        aL = self.addressList()
        for index, i in enumerate(aL):
            if i.split('\\')[-1] == str(targetID):
                targetIndex = index
                break
                
        _, imageAddListData = self.imageAddress()
        imageListPre = imageAddListData[targetIndex]
        
        imageList = []
        for i in imageListPre:
            imageItem = np.asarray(Image.open(i), dtype = np.float32)
            if imageItem.shape[0] == 50 and imageItem.shape[1] == 50 and imageItem.shape[2] == 3:
                imageList.append(i)
            else:
                continue
        
        testData = np.zeros((len(imageList), 50, 50, 3), dtype = np.float32)
        testLabel = np.zeros((len(imageList), ), dtype = np.float32)
        testLocation = []
        for index, i in enumerate(imageList):
            imageItem = np.asarray(Image.open(i), dtype = np.float32)
            testData[index, :, :, :] = np.asarray(Image.open(i), dtype = np.float32)
            testLabel[index] = float(int(i.split('.')[-2][-1]))
            x, y, _ = self.extractInformation(i)
            testLocation.append([y, x])
            
        testData = testData/255.
        testLocation = np.array(testLocation, dtype = np.int32)
        return imageList, testData, testLabel, testLocation
    
#------------------------------------------------------------------------------------------------------------------------------
#-----------Deal with code for presentation------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------

    def presentation4Train(self, numImage):
        '''
        Given the number as numImage, numImage iDs could be randomly picked and show the correspond slide images in sub-plots.
        '''
        aL = self.addressList()
        iD = [i.split('\\')[-1] for i in aL]
        
        idx = np.arange(len(aL))
        np.random.shuffle(idx)
        pickedID = [iD[i] for i in list(idx[0:numImage])]
        
        _, imageAddListData = self.imageAddress()
        fig, axes = plt.subplots(numImage, 2, figsize = (numImage*10, 20))
        fig.subplots_adjust(wspace = 0., hspace = 0.2)
        
        count = 0
        for index, i in enumerate(imageAddListData):
            if iD[index] in pickedID:
                image, mask = self.imageInformationSingle(i)
                imageArray = np.asarray(image, dtype = np.uint8)
                maskArray = np.asarray(mask, dtype = np.uint8)
                ax = axes[count]
                ax[0].imshow(imageArray)
                ax[0].set_title('Original Image for ID: {0}'.format(iD[index]))
                ax[1].imshow(maskArray)
                ax[1].set_title('Mask Image for ID: {0}'.format(iD[index]))
                count += 1
                
        plt.show()
        
#------------------------------------------------------------------------------------------------------------------------------
#-----------Deal with code for testing-----------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------
    def processPredLabel(self, predLabel):
        '''
        Given the output from trained classifier, predicted label would be calculated based on this output.
        '''
        res = np.argmax(predLabel, axis = 1)
        return res
    
    def maskPortion4imagePortionTest(self, label, shape = [50, 50, 3]):
        '''
        Return a RGB mask as shape based on the label.
        If label is 0, the mask could be white, else, the mask could be black.
        '''
        if label == 0:
            maskPortion = np.ones((shape[0], shape[1], shape[2]), dtype = np.float32) * 255.0
        else:
            maskPortion = np.zeros((shape[0], shape[1], shape[2]), dtype = np.float32) * 255.0
        return maskPortion
    
    def imageInformationSingleTest(self, testLocation, label, imageShape):
        '''
        Given the predicted labels of image portions from slide image for a patient ID.
        masks were produced based on each label of these and puzzled in original location based on coodinates.
        Return maskHolder is a mask for slide image for specific patient iD.
        '''
        maskHolder = np.ones((imageShape[0], imageShape[1], 3), dtype = np.float32)*255.0
        
        for i in range(testLocation.shape[0]):
            x, y, labelIndex = testLocation[i, 1], testLocation[i, 0], label[i]
            maskPortion = self.maskPortion4imagePortion(label = labelIndex, shape = [50, 50, 3])
            maskHolder[y:y+50, x:x+50, :] = maskPortion
        
        maskHolder = Image.fromarray(np.uint8(maskHolder), 'RGB')
        return maskHolder
    
    def accuracyItem(self, binaryLabel, testLabel):
        '''
        Given the predicted and original labels of image portions from a patient ID.
        Accuracy in percentage would be calculated between them.
        '''
        oriLabel = np.int32(testLabel)
        predLabel = binaryLabel
        res = float(np.sum(oriLabel == predLabel)/testLabel.shape[0])
        return res
    
    def mAP(self, maskOriginal, maskPredict):
        '''
        Given the predicted and original mask of slide image from a patient ID.
        mAP could be defined and calculated for measuring degree of overlaid between them.
        '''
        ori = np.asarray(maskOriginal, dtype = np.float32)[:, :, 0]
        pred = np.asarray(maskPredict, dtype = np.float32)[:, :, 0]
        ori4MAP = np.int32(ori == 0)
        pred4MAP = np.int32(pred == 0)
        
        res = ori4MAP + pred4MAP
        combine = np.float32(res == 2)
        union = np.float32(res > 0)
        res = float(np.sum(combine)/np.sum(union))
        return res
    
    def testItem(self, index, model):
        '''
        Given the index of the iD of patient in testing case and trained classifier, original slide image and mask and predicted mask, accuracy for classifying the image portions into binary labels and mAP between original mask and predicted mask would be calculated and returned.
        '''
        imageList, testData, testLabel, testLocation = self.prepareTestData(index)
        
        predLabel = model.predict(testData)
        #[predLabel, _] = model(testData)
        binaryLabel = self.processPredLabel(predLabel)
        
        image, maskOriginal = self.imageInformationSingle(imageList)
        imageShape = np.asarray(image).shape
        maskPredict = self.imageInformationSingleTest(testLocation, binaryLabel, imageShape)
        accuracy = self.accuracyItem(binaryLabel, testLabel)
        mAPValue = self.mAP(maskOriginal, maskPredict)
        return image, maskOriginal, maskPredict, accuracy, mAPValue
    
    def convertRGBCode(self, maskOriginal, maskPredict):
        '''
        Given the original and predicted mask, a blended mask image in PIL.Image where combine both maskOriginal, maskPredict but show them in different colors(Red for predicted and blue for original) would be returned.
        '''
        ori = np.asarray(maskOriginal, dtype = np.float32)
        pred = np.asarray(maskPredict, dtype = np.float32)
        (yOri, xOri, zOri) = np.where(ori == 0.)
        (yPred, xPred, zPred) = np.where(pred == 0.)
        
        for i in range(yOri.shape[0]):
            if zOri[i] == 0.:
                ori[yOri[i], xOri[i], zOri[i]] = 0.
            elif zOri[i] == 1.:
                ori[yOri[i], xOri[i], zOri[i]] = 0.
            else:
                ori[yOri[i], xOri[i], zOri[i]] = 255.
                
        for i in range(yPred.shape[0]):
            if zPred[i] == 0.:
                pred[yPred[i], xPred[i], zPred[i]] = 255.
            elif zPred[i] == 1.:
                pred[yPred[i], xPred[i], zPred[i]] = 0.
            else:
                pred[yPred[i], xPred[i], zPred[i]] = 0.
     , a blended
        imageOri = Image.fromarray(np.uint8(ori), 'RGB')
        imagePred = Image.fromarray(np.uint8(pred), 'RGB')
        imageBlend = Image.blend(imageOri, imagePred, 0.5)
        return imageBlend
    
    def Test(self, modelAddress, numShow = 3):
        '''
        Given the address of trained classifier and number of testing cases:numShow intend to plot.
        Average predicting accuracy and mAP for each testing iD would be calculated.
        Also, numShow testing cases would be randomly picked and plot their slide images and blended masks in designed subplots.
        '''
        dfTest = pd.read_csv(os.path.join(os.getcwd(), 'dataStore', 'testData', 'testiD.csv'))
        testiD = dfTest['iD'].tolist()
        
        model = load_model(modelAddress)
        imageList, maskOriList, maskPredList, accuracyList, mAPValueList = [], [], [], [], []
        for index, i in enumerate(testiD):
            image, maskOriginal, maskPredict, accuracy, mAPValue = self.testItem(index, model)
            imageList.append(image)
            maskOriList.append(maskOriginal)
            maskPredList.append(maskPredict)
            accuracyList.append(accuracy)
            mAPValueList.append(mAPValue)
            
        fig, axes = plt.subplots(numShow, 2, figsize = (numShow*10, 20))
        fig.subplots_adjust(wspace = 0., hspace = 0.2)
            
        idx = np.arange(len(testiD))
        np.random.shuffle(idx)
        targetID = [testiD[i] for i in list(idx[0:numShow])]
        targetIndex = list(idx[0:numShow])
        targetAccuracy = [accuracyList[i] for i in list(idx[0:numShow])]
        targetmAPValue = [mAPValueList[i] for i in list(idx[0:numShow])]
        for index, i in enumerate(targetID):
            image = imageList[targetIndex[index]]
            maskBlend = self.convertRGBCode(maskOriList[targetIndex[index]], maskPredList[targetIndex[index]])
            ax = axes[index]
            ax[0].imshow(image)
            ax[0].set_title('Original Image for ID: {0}, accuracy: {1:.2%}, mAP value: {2:.4}'.format(i, targetAccuracy[index], targetmAPValue[index]))
            ax[1].imshow(maskBlend)
            ax[1].set_title('Blend Image for ID: {0}, where red for prediction and blue for original'.format(i))
            
        print("The testing result could be shown as:\n")
        print("The average mAP value is {0:.4}, the average accuracy is {1:.2%}".format(statistics.mean(mAPValueList), statistics.mean(accuracyList)))
        print("The stdev of mAP value is {0:.4}, the stdev of accuracy is {1:.2%}".format(statistics.stdev(mAPValueList), statistics.stdev(accuracyList)))
                 
        return       