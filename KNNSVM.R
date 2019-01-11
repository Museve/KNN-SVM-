#Data Set Information:
# The data set can be gotten from the ICU machine learning website
  
# There are 9 predictors, all quantitative, and a binary dependent
#variable, indicating the presence or absence of breast cancer. 
#The predictors are anthropometric data and parameters which can
#be gathered in routine blood analysis. 
#Prediction models based on these predictors, if accurate, 
#can potentially be used as a biomarker of breast cancer.

#Attribute Information:
  
  ##Quantitative Attributes: 
#Age (years) 
#BMI (kg/m2) 
#Glucose (mg/dL) 
#Insulin (µU/mL) 
#HOMA 
#Leptin (ng/mL) 
#Adiponectin (µg/mL) 
#Resistin (ng/mL) 
#MCP-1(pg/dL) 

#Labels: 
# 1=Healthy controls 
# 2=Patients
# we will classify the data based on the response variable which has 
# two levels,1 and 2. 1 indicates presence of breast cancer, 2 denotes abscence


# KNN CLASSIFICATION
data=Cancerdata
attach(data)
rm(fff)
#model training
library(caTools)
data$Classification=as.factor(data$Classification)
ind=sample.split(data$Classification,SplitRatio = 0.8)
train=subset(data,ind==T)

test=subset(data,ind==F)
library(caret)
set.seed(123)
model <- train((Classification)~., data =train, method = "knn",
  trControl = trainControl("cv", number = 10),
  preProcess = c("center","scale"),
  tuneLength = 20
)
# Plot model accuracy vs different values of k
plot(model)
model$bestTune
knnpred=predict(model,test)
head(knnpred)

#[1] 2 1 2 1 1 2
#Levels: 1 2
# confusion matrix
cm=table(Predicted=knnpred,Actual=test$Classification)
confusionMatrix(cm)
#Confusion Matrix and Statistics

#         Actual
#Predicted 1 2
#        1 7 4
#        2 3 9

#Accuracy : 0.6957          
#95% CI : (0.4708, 0.8679)
#No Information Rate : 0.5652          
#P-Value [Acc > NIR] : 0.1462          

#Kappa : 0.3878          
#Mcnemar's Test P-Value : 1.0000          

#Sensitivity : 0.7000          
#Specificity : 0.6923          
#Pos Pred Value : 0.6364          
#Neg Pred Value : 0.7500          
#Prevalence : 0.4348          
#Detection Rate : 0.3043          
#Detection Prevalence : 0.4783          
#Balanced Accuracy : 0.6962          

#'Positive' Class : 1 
#misclassification error
mean(knnpred!=test$Classification)
#[1] 0.3043478

#the error 30.43% is too large for KNN classifier.


#SVM CLASSIFIER
# load necesssary library
library(e1071)
#Linear svm
modelsvm <- train(
  Classification ~., data = train, method = "svmLinear",
  trControl = trainControl("cv", number = 10),
  tuneGrid = expand.grid(C = seq(0, 2, length = 20)),
  preProcess = c("center","scale")
)
# Plot model accuracy vs different values of Cost
plot(modelsvm)
modelsvm$bestTune
     C
#11 1.052632
#predicting test data
svmpred=predict(modelsvm,test)
cmsvm=table(Predicted=svmpred,Actual=test$Classification)
confusionMatrix(cmsvm)
#Confusion Matrix and Statistics

 #        Actual
#Predicted  1  2
#        1  7  2
#        2  3 11

#Accuracy : 0.7826         
#95% CI : (0.563, 0.9254)
#No Information Rate : 0.5652         
#P-Value [Acc > NIR] : 0.02627        

#Kappa : 0.5525         
#Mcnemar's Test P-Value : 1.00000        

#Sensitivity : 0.7000         
#Specificity : 0.8462         
#Pos Pred Value : 0.7778         
#Neg Pred Value : 0.7857         
#Prevalence : 0.4348         
#Detection Rate : 0.3043         
#Detection Prevalence : 0.3913         
#Balanced Accuracy : 0.7731         

#'Positive' Class : 1              
#misclassification error
mean(svmpred!=test$Classification)
#[1] 0.2173913
#this linear classifier is better than KNN the error is 21.73%
# we fit radial classifier
modelradial <- svm(Classification~., train
)
# Plot model accuracy vs different values of Cost
radpred=predict(modelradial,test)
tabrad=table(predicted=radpred,actual=test$Classification)
confusionMatrix(tabrad)
#Confusion Matrix and Statistics

           actual
#predicted  1  2
#        1  7  2
#        2  3 11

Accuracy : 0.7826         
#95% CI : (0.563, 0.9254)
#No Information Rate : 0.5652         
#P-Value [Acc > NIR] : 0.02627        

#Kappa : 0.5525         
#Mcnemar's Test P-Value : 1.00000        

#Sensitivity : 0.7000         
#Specificity : 0.8462         
#Pos Pred Value : 0.7778         
#Neg Pred Value : 0.7857         
#Prevalence : 0.4348         
#Detection Rate : 0.3043         
#Detection Prevalence : 0.3913         
#Balanced Accuracy : 0.7731         

#'Positive' Class : 1              

#Accuracy of the radial svm is 78%.it is still a better
#predictor than KNN
#fitting ploynomial svm
modelpoly <- train(Classification ~., data = train, 
method = "svmPoly",trControl = trainControl("cv", number = 10),
  tunelength=25,preProcess = c("center","scale"))
plot(modelpoly)
modelpoly
#Support Vector Machines with Polynomial Kernel 

#93 samples
#9 predictor
#2 classes: '1', '2' 

#Pre-processing: centered (9), scaled (9) 
#Resampling: Cross-Validated (10 fold) 
#Summary of sample sizes: 83, 84, 84, 83, 84, 84, ... 
#Resampling results across tuning parameters:
  
#  degree  scale  C     Accuracy   Kappa     
#1       0.001  0.25  0.5488889  0.00000000
#1       0.001  0.50  0.5488889  0.00000000
#1       0.001  1.00  0.5488889  0.00000000
#1       0.010  0.25  0.5488889  0.00000000
#1       0.010  0.50  0.5488889  0.00000000
#1       0.010  1.00  0.5488889  0.00000000
#1       0.100  0.25  0.6577778  0.30710625
#1       0.100  0.50  0.6755556  0.34785178
#1       0.100  1.00  0.6855556  0.37203297
#2       0.001  0.25  0.5488889  0.00000000
#2       0.001  0.50  0.5488889  0.00000000
#2       0.001  1.00  0.5488889  0.00000000
#2       0.010  0.25  0.5488889  0.00000000
#2       0.010  0.50  0.5488889  0.00000000
#2       0.010  1.00  0.5500000  0.04407484
#2       0.100  0.25  0.6866667  0.37136752
#2       0.100  0.50  0.7288889  0.46290062
#2       0.100  1.00  0.7200000  0.43690717
#3       0.001  0.25  0.5488889  0.00000000
#3       0.001  0.50  0.5488889  0.00000000
#3       0.001  1.00  0.5488889  0.00000000
#3       0.010  0.25  0.5488889  0.00000000
#3       0.010  0.50  0.5488889  0.01449631
#3       0.010  1.00  0.6666667  0.32977486
#3       0.100  0.25  0.7300000  0.45518317
#3       0.100  0.50  0.7511111  0.49823195
#3       0.100  1.00  0.6877778  0.38698243

#Accuracy was used to select the optimal model using
#the largest value.
#The final values used for the model were degree = 3, scale
#= 0.1 and C = 0.5.

modelpoly$bestTune
#degree scale   C
#26      3   0.1 0.5
