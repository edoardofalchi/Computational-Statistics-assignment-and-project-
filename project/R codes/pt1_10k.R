# PART 1
#######################################
#models comparison evaluation deploying a smaller dataset simulated starting from the original dataset (10k observations instead of 30k)
##########################################

#Loading the required libraries
library(data.table)
library(ggplot2)
library(dplyr)
library(caret)
library(pROC)
library(ROCR)
library(MASS)
library(dummies)
library(class)
library(e1071)
library(plyr)
library(randomForest)
library(fishmethods)
library(Hmisc)
library(CustomerScoringMetrics)
library(kernlab)
library(nnet)
library(parallel)
library(doParallel)#speed up model building by using parallel computing
set.seed(321)

#load data
data <- read.csv("default of credit card clients.csv")

## Droping the unnecessary variable/column
data<-data[,-1]#There's no use of the column id in the analysis
##Looking for missing value
sum(is.na(data))#it is observed that there's no missing values
##Changing the name of the variable PAY_0 to PAY_1 and default.payment.next.month to target
names(data)[6]<-"PAY_1"
names(data)[24] <- "target"

##Transforming the variables SEX,MARRIAGE,EDUCATION and default payment into factors
df=as.data.frame(data)
df[c("SEX","MARRIAGE","EDUCATION","target","PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6")]= 
  lapply(df[c("SEX","MARRIAGE","EDUCATION","target","PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6")]
         ,function(x) as.factor(x))                                                                       
data=df
rm(df)

#There are some undocumented labels in the factor variables like EDUCATION and MARRIAGE.
#For example, the labels 0, 5 and 6 of EDUCATION are not documented clearly in the description of the dataset, so I merge these labels with the label 4 that implies qualification other than high school, graduate and university.
data$EDUCATION <- ifelse(data$EDUCATION == 0 |data$EDUCATION == 5 | data$EDUCATION == 6,
                         4, data$EDUCATION)
#Similarly, we merge  0 to 3 for MARRIAGE factor.
data$MARRIAGE = ifelse(data$MARRIAGE == 0, 3, data$MARRIAGE)


simulate_data <- function(N) {
  quantitative=data[,c(-2:-4,-6:-11,-24)]
  qualitative=data[,c(2:4,6:11,24)]
  df <- setNames(data.frame(matrix(nrow = N, ncol = length(colnames(data)))), 
                 colnames(data))
  
  
  for (i in 1:ncol(data)){
    ifelse(colnames(data)[i] %in% colnames(qualitative),
           df[,i]<-sample(as.numeric(names(table(data[,i]))), N, replace = TRUE, prob = as.data.frame(prop.table(table(data[,i])))$Freq),
           df[,i]<-remp(N, as.numeric(data[,i]))#with remp() an empirical probability distribution is formed from empirical data with each observation having 1/T probabililty of selection, where T is the number of data points
    )
  }
  
  ##Transforming qualitative variables SEX,MARRIAGE,EDUCATION and default payment into factors
  df[c("SEX","MARRIAGE","EDUCATION","target","PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6")]<- 
    lapply(df[c("SEX","MARRIAGE","EDUCATION","target","PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6")]
           ,function(x) as.factor(x))                                                                          
  
  return(df)
}

sim<-simulate_data(10000)

#distribution of the classification variable for the original and simulated datasets
hist_data <- setNames(as.data.frame(data$target), "default")
hist_sim_data <- setNames(as.data.frame(sim$target), "default")
hist_data$Data <- "Original"
hist_sim_data$Data <- "Simulated"
hist_comb <- rbind(hist_data, hist_sim_data)
default_status <- ggplot(hist_comb, aes(default, group = Data)) + 
  geom_bar(aes(y = ..prop..), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  ylab("relative frequencies") +
  facet_grid(~Data)+
  theme(aspect.ratio = 1)
options(repr.plot.width=8, repr.plot.height=3)
default_status

#Let's split the combined dataframe into two parts.
#One is training set, consists of 80% of the data, on which the model(s) will be trained and the other one is test set, consists of remaining 20% of the data, on which the model(s) will be validated.
#Splitting the data into test and train sets in 80:20 ratio
ind=sample(nrow(sim),0.8*nrow(sim),replace = F)
train=sim[ind,]
test=sim[-ind,]

#Creating empty vectors for further comparisons
acc.train=numeric()
sens.train=numeric() 
spec.train=numeric()
auc.train=numeric()
acc.test=numeric()
sens.test=numeric() 
spec.test=numeric()
auc.test=numeric()

#A. LOGISTIC ###############################################
#fitting a logistic model
model.logit=glm(target~.,data=train,family="binomial")

#For further analysis we might also want to
#restrict varbiables to the first 10 by importance based on classification model
imp <- varImp(model.logit)
impDF<-data.frame(imp[1])
top10var<- rownames(impDF)[order(imp$importance, decreasing=TRUE)[1:10]]
frm<-as.formula(paste("target ~ ", paste(top10var, collapse="+")))


#Making prediction for the train set and test set
pred.logit=predict(model.logit,type="response",newdata = test)
pred.test=ifelse(pred.logit>0.5,"1","0")
pred.train=ifelse(predict(model.logit,type="response",newdata = train)>0.5,"1","0")

#Calculate error rate for both train and test set
conf.train<-confusionMatrix(as.factor(pred.train),(train$target),positive="1")
conf.test<-confusionMatrix(as.factor(pred.test),(test$target),positive="1")
acc.train[1]<-conf.train$overall['Accuracy']
sens.train[1]<-(conf.train$byClass)[1]
spec.train[1]<-(conf.train$byClass)[2]
acc.test[1]<-conf.test$overall['Accuracy']
sens.test[1]<-(conf.test$byClass)[1]
spec.test[1]<-(conf.test$byClass)[2]
#Ploting ROC curve and AUC for test and train set
par(mfrow=c(1,2))
par(pty="s")
#For training set
roc_log.train<-roc(train$target,model.logit$fitted.values,plot=T,col="#69b3a2",print.auc=T,legacy.axes=TRUE,percent = T,
    xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Train Set")
#For test set
roc_log.test<-roc(test$target,pred.logit,plot=T,col="navyblue",print.auc=T,legacy.axes=TRUE,percent = T,
    xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Test Set")

auc.train[1]=auc(train$target,model.logit$fitted.values)
auc.test[1]=auc(test$target,pred.logit)





# B. LDA #############################################

#Besides the formula and training data, one more parameter prior is passed to the function lda().
#prior is a vector specifying the prior probabilities of class membership. We will use the proportion of the classes in our dataset as our input.
#prior class proportion
prop.table(table(data$target))
#fitting a LDA model
model.lda=lda(target~.,data=train,prior=c(prop.table(table(data$target))))

#making prediction for both train and test sets
pred.lda.test=predict(model.lda,test)
pred.lda.prob=pred.lda.test$posterior[,2]
pred.lda.train=predict(model.lda,train)


#calculate the error rate for both train and test sets
conf.train<-confusionMatrix(data=pred.lda.train$class,reference=(train$target),positive="1")
conf.test<-confusionMatrix(data=pred.lda.test$class,reference=(test$target),positive="1")
acc.train[2]<-conf.train$overall['Accuracy']
sens.train[2]<-(conf.train$byClass)[1]
spec.train[2]<-(conf.train$byClass)[2]
acc.test[2]<-conf.test$overall['Accuracy']
sens.test[2]<-(conf.test$byClass)[1]
spec.test[2]<-(conf.test$byClass)[2]


#Ploting ROC curve and AUC for test and train set
par(mfrow=c(1,2))
par(pty="s")
#For training set
roc_lda.train<-roc(response=train$target,predictor=pred.lda.train$posterior[,2],plot=T,col="#69b3a2",print.auc=T,legacy.axes=TRUE,percent = T,
    xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Train Set")
#For test set
roc_lda.test<-roc(response=test$target,predictor=pred.lda.prob,plot=T,col="navyblue",print.auc=T,legacy.axes=TRUE,percent = T,
    xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Test Set")

auc.train[2]=auc(train$target,pred.lda.train$posterior[,2])
auc.test[2]=auc(test$target,pred.lda.prob)



#C. K-NEAREST NEIGHBOUR ###############################################

#Any variables that are on a large scale will have a much larger
#effect on the distance between the observations, and hence on the KNN
#classifier, than variables that are on a small scale.
#Let's standardize the data, then all variables will be on a comparable scale.
standardized.X=scale(sim[,c(-2:-4,-6:-11,-24)])
quali.dummy=dummy.data.frame(sim[,c(2:4,6:11)])
#Merging the normalized data and one-hot encoded dummies
target<-sim$target
data.knn=cbind(standardized.X,quali.dummy,target)
train.knn=data.knn[ind,]
test.knn=data.knn[-ind,]

model.list=list()#empty list
v=numeric()

for(i in 1:30){
  
  model.list[[i]]=knn(train.knn[,c(-83)],test.knn[,c(-83)],train.knn[,c(83)] ,k=i)
  tab=table(model.list[[i]],test.knn[,c(83)])
  v[i]=sum(diag(tab)/sum(tab))
  
}
which.max(v)#index of the K that maximizes accuracy
plot(1:30,v,type="b",xlab="k",ylab="accuracy",main="Elbow plot",font.main=2,col="steelblue3",lwd=4)
abline(v=which.max(v),col="orange")

#Prediction and calculating error rate on test set
knn.pred=knn(train.knn[,c(-83)],test.knn[,c(-83)],train.knn[,c(83)] ,k=which.max(v))#Best model in terms of accuracy is when k=12
conf.test<-confusionMatrix(knn.pred ,as.factor(test.knn[,c(83)]),positive="1")

#Prediction and calculating error rate on training set
model.knn.train=knn3(train.knn[,c(-83)],as.factor(train.knn[,c(83)]),k=which.max(v))
conf.train<-confusionMatrix(predict(model.knn.train,train.knn[,c(-83)],type = "class"),as.factor(train.knn[,83]),positive="1")

acc.train[3]<-conf.train$overall['Accuracy']
sens.train[3]<-(conf.train$byClass)[1]
spec.train[3]<-(conf.train$byClass)[2]
acc.test[3]<-conf.test$overall['Accuracy']
sens.test[3]<-(conf.test$byClass)[1]
spec.test[3]<-(conf.test$byClass)[2]
#Ploting ROC curve and AUC for test and train set
par(mfrow=c(1,2))
par(pty="s")
roc_knn.train<-roc(train.knn[,c(83)],predict(model.knn.train,train.knn[,c(-83)],type="prob")[,2],plot=T,col="#69b3a2",print.auc=T,legacy.axes=TRUE,
    percent = T,xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Train Set")
roc_knn.test<-roc(test.knn[,c(83)],predict(model.knn.train,test.knn[,c(-83)],type="prob")[,2],plot=T,col="navyblue",print.auc=T,legacy.axes=TRUE,percent = T,
    xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Test Set")

auc.train[3]=auc(train.knn[,c(83)],predict(model.knn.train,train.knn[,c(-83)],type="prob")[,2])
auc.test[3]=auc(test.knn[,c(83)],predict(model.knn.train,test.knn[,c(-83)],type="prob")[,2])



#D. RANDOM FOREST #############
data.rf<-cbind(standardized.X,sim[,c(2:4,6:11)],target)
train.rf<-data.rf[ind,]
test.rf<-data.rf[-ind,]

no_cores <- detectCores()-1  #Calculate the number of cores
cl <- makePSOCKcluster(no_cores)#create the cluster for caret to use
registerDoParallel(cl)
#Tuning Hyperparameters for RF
param = trainControl(method = "cv",number = 5 , allowParallel = T,classProbs = T,summaryFunction = twoClassSummary)
RandomForest_Fit = caret::train(target~.,train.rf, method = 'rf',trControl = param, tuneGrid=data.frame(.mtry = seq(5,20, by=5)))
stopCluster(cl)
registerDoSEQ()

#Model Fitting based on the Grid Search
model.rf=randomForest(target~., data = train.rf, mtry =RandomForest_Fit$bestTune[[1]],maxnodes=30,importance=T)

#Prediction and calculating error rate for test and training sets
#for test set
pred.rf= predict(model.rf,newdata = test.rf[,-24],probability=T)
pred.rf.prob= predict(model.rf,newdata = test.rf[,-24],type="prob")
conf.test<-confusionMatrix(pred.rf ,test.rf$target,positive="1")

#for training set
pred.rf.train= predict(model.rf,newdata = train.rf[,-24],probability=T)
pred.rf.train.prob=predict(model.rf,newdata = train.rf[,-24],type="prob")
conf.train=confusionMatrix(pred.rf.train,train.rf$target, positive="1")

acc.train[4]<-conf.train$overall['Accuracy']
sens.train[4]<-(conf.train$byClass)[1]
spec.train[4]<-(conf.train$byClass)[2]
acc.test[4]<-conf.test$overall['Accuracy']
sens.test[4]<-(conf.test$byClass)[1]
spec.test[4]<-(conf.test$byClass)[2]
#ROC curve and AUC value for both train and test set
par(mfrow=c(1,2))
par(pty="s")
roc_rf.train<-roc(train.rf$target,(pred.rf.train.prob[,2]),plot=T,
    col="#69b3a2",print.auc=T,legacy.axes=TRUE,percent = T,xlab="False Positive percentage",
    ylab="True Positive percentage",lwd=5,main="Train Set")
roc_rf.test<-roc(test.rf$target,pred.rf.prob[,2],plot=T,col="navyblue",print.auc=T,legacy.axes=TRUE,percent = T,
    xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Test Set")

auc.train[4] = auc(train.rf$target,(pred.rf.train.prob[,2]))
auc.test[4] = auc(test.rf$target,pred.rf.prob[,2])




#CLASSIFICATION EVALUATION ############################
classification.eval=data.frame(Model=c("Logistic","LDA","KNN","Random forest"),Accuracy_train=acc.train,
                               Sensitivity_train=sens.train, Specifivity_train=spec.train, AUC_train=auc.train,
                               Accuracy_test=acc.test, Sensitivity_test=sens.test, Specificity_test=spec.test, AUC_test=auc.test)
classification.eval

