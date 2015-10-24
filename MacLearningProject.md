# Practical Machine Learning Prediction Project
Sam  
Friday, October 23, 2015  

##Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

##Data Processing
The training data for this project are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
The test data are available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv


```r
setInternet2(TRUE)
download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = "pml-training.csv")
download.file(url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = "pml-testing.csv")
training <- read.csv('pml-training.csv')
testing <- read.csv('pml-testing.csv')

dim(training)
```

```
## [1] 19622   160
```

```r
dim(testing)
```

```
## [1]  20 160
```

##Prepare The Datasets




```r
library(survival)
library(splines)
library(plyr)
library(doParallel)
```

```
## Loading required package: foreach
## Loading required package: iterators
## Loading required package: parallel
```

```r
library(e1071)
library(gbm)
```

```
## Loading required package: lattice
## Loaded gbm 2.1.1
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
```

```r
library(ggplot2)
library(caret)
```

```
## 
## Attaching package: 'caret'
## 
## The following object is masked from 'package:survival':
## 
##     cluster
```

```r
training <- training[, 6:dim(training)[2]]

treshold <- dim(training)[1] * 0.95
#Remove columns with more than 95% of NA or "" values
goodColumns <- !apply(training, 2, function(x) sum(is.na(x)) > treshold  || sum(x=="") > treshold)

training <- training[, goodColumns]

badColumns <- nearZeroVar(training, saveMetrics = TRUE)

training <- training[, badColumns$nzv==FALSE]

training$classe = factor(training$classe)
```

##Splitting data
We separate our training data into a training set and a validation set so that we can validate our model


```r
#Partition rows into training and validation
inTrain <- createDataPartition(training$classe, p = 0.6)[[1]]
crossv <- training[-inTrain,]
training <- training[ inTrain,]
inTrain <- createDataPartition(crossv$classe, p = 0.75)[[1]]
crossv_test <- crossv[ -inTrain,]
crossv <- crossv[inTrain,]

testing <- testing[, 6:dim(testing)[2]]
testing <- testing[, goodColumns]
testing$classe <- NA
testing <- testing[, badColumns$nzv==FALSE]
```
 


##Training Model
Now, let's train the model, you can see some information of mod1 

```r
mod1<-randomForest(classe ~ ., data = training, importance = TRUE, ntrees = 10)

mod1
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training, importance = TRUE,      ntrees = 10) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.34%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3347    0    0    0    1 0.0002986858
## B    6 2269    4    0    0 0.0043878894
## C    0    8 2046    0    0 0.0038948393
## D    0    0   16 1912    2 0.0093264249
## E    0    0    0    3 2162 0.0013856813
```

##The Accuracy of my model

```r
pred1 <- predict(mod1, crossv) 

#show confusion matrices
confusionMatrix(pred1, crossv$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    0 1139    6    0    0
##          C    0    0 1020    7    0
##          D    0    0    0  958    2
##          E    0    0    0    0 1080
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9975          
##                  95% CI : (0.9958, 0.9986)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9968          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   0.9942   0.9927   0.9982
## Specificity            1.0000   0.9987   0.9986   0.9996   1.0000
## Pos Pred Value         1.0000   0.9948   0.9932   0.9979   1.0000
## Neg Pred Value         1.0000   1.0000   0.9988   0.9986   0.9996
## Prevalence             0.2844   0.1935   0.1743   0.1639   0.1838
## Detection Rate         0.2844   0.1935   0.1733   0.1628   0.1835
## Detection Prevalence   0.2844   0.1945   0.1745   0.1631   0.1835
## Balanced Accuracy      1.0000   0.9994   0.9964   0.9962   0.9991
```
Random Forest prediction was far better than other models.  The confusion matrix created gives an accuracy of 99.6%.

Out of sample error was calculated has 99.7% accuracy on the validation set.


##Conclusion
The Random Forest (RF) method is the best prediction model for this dataset. The Confusion Matrix achieved 99.6% accuracy. The Out of Sample Error achieved 99.7%. Based on this conclusion RF will be used for the finall calculations.

###Submission. (using COURSERA provided code) 

```r
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}



predanswers <- predict(mod1, newdata=testing)
predanswers
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

```r
pml_write_files(as.character(predanswers))
```
