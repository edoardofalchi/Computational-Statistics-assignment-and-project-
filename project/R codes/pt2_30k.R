# PART 2
#######################################
#models comparison evaluation deploying the original and larger dataset (30k observations)
##########################################
set.seed(333)



#Data partitioning
ind=sample(nrow(data),0.8*nrow(data),replace = F)
train=data[ind,]
test=data[-ind,]


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

standardized.X=scale(data[,c(-2:-4,-6:-11,-24)])
quali.dummy=dummy.data.frame(data[,c(2:4,6:11)])
#Merging the normalized data and one-hot encoded dummies
target<-data$target
data.knn=cbind(standardized.X,quali.dummy,target)
train.knn=data.knn[ind,]
test.knn=data.knn[-ind,]

no_cores <- detectCores()-1  #Calculate the number of cores
cl <- makePSOCKcluster(no_cores)#create the cluster for caret to use
registerDoParallel(cl)

knn_fit<-caret::train(target ~ .,data=train.knn,method = 'knn',trControl =trainControl(method="repeatedcv",repeats = 3))
stopCluster(cl)
registerDoSEQ()               

#Prediction and calculating error rate on test set
knn.pred=knn(train.knn[,c(-83)],test.knn[,c(-83)],train.knn[,c(83)] ,k=knn_fit$bestTune[[1]])#Best model in terms of accuracy is when k=9
conf.test<-confusionMatrix(knn.pred ,as.factor(test.knn[,c(83)]),positive="1")

#Prediction and calculating error rate on training set
model.knn.train=knn3(train.knn[,c(-83)],as.factor(train.knn[,c(83)]),k=knn_fit$bestTune[[1]])
conf.train<-confusionMatrix(predict(model.knn.train,train.knn[,c(-83)],type = "class"),as.factor(train.knn[,83]),positive="1")

acc.train[3]<-conf.train$overall['Accuracy']
sens.train[3]<-(conf.train$byClass)[1]
spec.train[3]<-(conf.train$byClass)[2]
acc.test[3]<-conf.test$overall['Accuracy']
sens.test[3]<-(conf.test$byClass)[1]
spec.test[3]<-(conf.test$byClass)[2]
#Plotting ROC curve and AUC for test and train set
par(mfrow=c(1,2))
par(pty="s")
roc_knn.train<-roc(train.knn[,c(83)],predict(model.knn.train,train.knn[,c(-83)],type="prob")[,2],plot=T,col="#69b3a2",print.auc=T,legacy.axes=TRUE,
                   percent = T,xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Train Set")
roc_knn.test<-roc(test.knn[,c(83)],predict(model.knn.train,test.knn[,c(-83)],type="prob")[,2],plot=T,col="navyblue",print.auc=T,legacy.axes=TRUE,percent = T,
                  xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Test Set")

auc.train[3]=auc(train.knn[,c(83)],predict(model.knn.train,train.knn[,c(-83)],type="prob")[,2])
auc.test[3]=auc(test.knn[,c(83)],predict(model.knn.train,test.knn[,c(-83)],type="prob")[,2])


#D. RANDOM FOREST #############
target=recode_factor(target,'0'="no",'1'="yes")
data.rf<-cbind(standardized.X,data[,c(2:4,6:11)],target)
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
conf.test<-confusionMatrix(pred.rf ,test.rf$target,positive="yes")

#for training set
pred.rf.train= predict(model.rf,newdata = train.rf[,-24],probability=T)
pred.rf.train.prob=predict(model.rf,newdata = train.rf[,-24],type="prob")
conf.train=confusionMatrix(pred.rf.train,train.rf$target, positive="yes")

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
roc_rf.test<-roc(test.rf$target,pred.rf.prob[,2],plot=T,col="navyblue",print.auc=T,percent=T,
                 xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Test Set")

auc.train[4] = auc(train.rf$target,(pred.rf.train.prob[,2]))
auc.test[4] = auc(test.rf$target,pred.rf.prob[,2])


#CLASSIFICATION EVALUATION ############################
classification.eval_30k=data.frame(Model=c("Logistic","LDA","KNN","Random forest"),Accuracy_train=acc.train,
                               Sensitivity_train=sens.train, Specifivity_train=spec.train, AUC_train=auc.train,
                               Accuracy_test=acc.test, Sensitivity_test=sens.test, Specificity_test=spec.test, AUC_test=auc.test)
classification.eval_30k

# plot ROC curves together in one single plot
rocs_train<- function(){
  plot(roc_log.train, col = "green", lty=1, main = "ROCs train set")
  lines(roc_lda.train, col = "red", lty=2)
  lines(roc_knn.train, col = "orange", lty=3)
  lines(roc_rf.train, col = "blue", lty=4)
  legend(57,40,legend=c("Logistic", "LDA","KNN","Random Forest"),
         col=c("green", "red", "orange","blue"),lty=1:4, cex=0.55)
}
rocs_test<- function(){
  plot(roc_log.test, col = "green", lty=1, main = "ROCs test set")
  lines(roc_lda.test, col = "red", lty=2)
  lines(roc_knn.test, col = "orange", lty=3)
  lines(roc_rf.test, col = "blue", lty=4)
  legend(57,40,legend=c("Logistic", "LDA","KNN","Random Forest"),
         col=c("green", "red", "orange","blue"),lty=1:4, cex=0.55)
}

par(mfrow=c(1,2))
#ROCS train set
rocs_train()
#ROCs test set
rocs_test()



#APPENDIX - CUMULATIVE GAINS CHART##############

#The graph is constructed with the cumulative number of cases (in descending order of probability) on the x-axis and the cumulative number of true positives on the y-axis
#True positives are those observations from the important class (here class 1) that are classified correctly.
#The bisector solid line is a reference line. For any given number of cases (the x-axis value), it represents the expected number of positives we would predict if we did not
#have a model but simply selected cases at random. It provides a benchmark against which we can see performance of the model.

par(mfrow=c(2,2))
#Cumulative Gain Chart for The Logit
cumGainsChart(as.numeric(pred.test), as.numeric(test$target), resolution = 1/10)
#Cumulative Gain Chart for The LDA
cumGainsChart((pred.lda.prob), (test$target), resolution = 1/10)
#Cumulative Gain Chart for KNN model
cumGainsChart((predict(model.knn.train,test.knn[,c(-83)],type="prob")[,2]), (test.knn[,c(83)]), resolution = 1/10)
#Cumulative Gain Chart for Random Forest 
cumGainsChart(pred.rf.prob[,2],test.rf$target, resolution = 1/10)


