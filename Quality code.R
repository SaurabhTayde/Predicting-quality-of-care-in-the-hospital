getwd()
setwd('G:/DSP/LogisticRegressionPractice/titanic/Logistic Regression Practice/Logistic Regression Practice/Quality')
quality = read.csv('quality.csv')

head(quality)

str(quality)

summary(quality)

dim(quality)

#No need to do following step as Pooracare is already in 0 and 1.
#This step is needed when dependent variable is not 0 and 1.

#quality$PoorCare = factor(quality$PoorCare, levels = c(0,1))

#Now we will check baseline model.
#It means what is the accuracy we are getting just by observing the dependent variable

table(quality$PoorCare)

#So PoorCare = 33, GoodCare = 98
#So baseline model gives accuracy of 98/(33+98) = 0.7480916
#74.8 %
#So we should achieve accuracy greater than this

#install.packages('caTools')

library(caTools)

set.seed(88)


split = sample.split(quality$PoorCare, SplitRatio = 0.8)
split

trainData = subset(quality, split == TRUE)
testData = subset(quality, split == F)

table(trainData$PoorCare)

table(testData$PoorCare)

nrow(trainData)

nrow(testData)

TrainModel1 = glm(PoorCare ~. , family = binomial, data = trainData)

summary(TrainModel1)

#quality$StartedOnCombination

#install.packages('fmsb')

#library(fmsb)

#install.packages('car')

library(car)

#install.packages('MASS')

library(MASS)


stepAIC(TrainModel1, direction = 'both')

#Use result of stepAIC and form a new model that will provide which coeff are important to the model

TrainModel2 = glm(formula = PoorCare ~ OfficeVisits + 
      AcuteDrugGapSmall, family = binomial, data = trainData)

dim(TrainModel2)

plot(TrainModel2)

summary(TrainModel2)

vif(TrainModel2)

#Let us call it predictTrain and use the predict function to make predictions using the model QualityLog.
#We will also use an argument called type="response" which gives us the probabilities.

#We should always predict on unseen observations but here we want to get the value of threshold,
#Hence the predictions on train data set only.

predictTrain = predict(TrainModel2, type = 'response')

predictTrain

summary(predictTrain)

tapply(predictTrain, trainData$PoorCare, mean)  #Sort by poorcare and then took the mean

#0         1 
#0.1696893 0.4909321 



#We find that for all of the true poor care cases, we predict an average probability of about 0.49.
#And for all of the true good care cases, we predict an average probability of about 0.17.
#This is good because it looks like we're predicting a higher probability for the actual poor care cases.


#Thresholding:
#We can convert the probabilities to predictions using what's called a threshold value, t.
#If the probability of poor care is greater than this threshold value, t,
#we predict poor quality care. But if the probability of poor care is less than the threshold value, t, then we predict good quality care.
#ROC curve helps to decide threshold value.




#Confusion Matrix:


#        FALSE		 TRUE
# 0    	  TN		    FP
# 1    	  FN		    TP


table(trainData$PoorCare, predictTrain > 0.5)

# FALSE TRUE
# 0    73    5
# 1    16   10

#Sensitivity = 0.40 (10/26)
#Specificity = 93.58 (73/78)


table(trainData$PoorCare, predictTrain > 0.7)

# FALSE TRUE
# 0    76    2
# 1    18    8

#Sensitivity = 30.76 (8/26)
#Specificity = 97.43 (76/78)


table(trainData$PoorCare, predictTrain > 0.2)

# FALSE TRUE
# 0    58   20
# 1     6   20

#Sensitivity = 20/26 = 76.92
#Specificity = 58/78 = 74.35

# We see that by increasing the threshold value, the model's sensitivity decreases and specificity increases
# while the reverse happens if the threshold value is decreased



#**********ROCR**********

install.packages('ROCR')
library(ROCR)

#Creating ROC curve:

# The first is the predictions we made with our model, which we called predictTrain.
# The second argument is the true outcomes of our data points,
# which in our case, is trainData$PoorCare.

ROCpred = prediction(predictTrain, trainData$PoorCare)

ROCpred


#Use the performance function which defines
#what we'd like to plot on the x and y-axes of our ROC curve.


ROCperf = performance(ROCpred, 'tpr', 'fpr')

plot(ROCperf)

# Add colors

plot(ROCperf, colorize = T)

#Now to add values on the curve using print.cutoff:

plot(ROCperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1))

#As we can see in the graph, we are not able to see some values due to overlappping
#So adjust these values using text adjust

plot(ROCperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))


# High Threshold :
# High specificity
# Low sensitivity


# Low Threshold:
# Low specificity
# High sensitivity


# TPR = TP /(TP + FN) =  Sensitivity
 
# FPR = FP/(FP + TN)  = 1 - Specificity


# The ROC curve always starts at the point (0, 0) i.e threshold of value 1
# This means at this threshold we will not catch any poor care cases(sensitivity of 0)
# but will correctly label of all the good care cases(FP = 0)


# The ROC curve always ends at the point (1,1) i.e threshold of value 0.
# This means at this threshold we will catch all the poor care cases(sensitivity of 1)
# But will incorrectly label of all the good care case as poor cases(FP = 1)

predicttest = predict(TrainModel2, type = 'response', newdata = testData)

predicttest

table(testData$PoorCare, predicttest>= 0.4)

#Accuracy

#21/27 * 100 = 77.77%


#Initial baseline model gives accuracy of 74.8% while logistic model gives accuracy of 77.77%



