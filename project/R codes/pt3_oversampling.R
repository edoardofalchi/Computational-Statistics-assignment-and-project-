# PART 3
#######################################
#models comparison evaluation after oversampling for better sensitivity
##########################################
set.seed(333)

prop.table(table(data$target))#imbalanced
over<-caret::upSample(data[,-24],data$target,yname="target")
prop.table(table(over$target))#now the dataset is balanced according to the dependent classification variable
ggplot(over, aes(x = target)) + 
  geom_bar()

#Data partitioning
ind_over=sample(nrow(over),0.8*nrow(over),replace = F)
train_over=over[ind_over,]
test_over=over[-ind_over,]


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
model.logit_over=glm(target~.,data=train_over,family="binomial")

#Making prediction for the train set and test set
pred.logit_ov=predict(model.logit_over,type="response",newdata = test_over)
pred.test_ov=ifelse(pred.logit_ov>0.5,"1","0")
pred.logit_train_ov=predict(model.logit_over,type="response",newdata = train_over)
pred.train_ov=ifelse(pred.logit_train_ov>0.5,"1","0")

#Calculate error rate for both train and test set
conf.train_ov<-confusionMatrix(as.factor(pred.train_ov),(train_over$target),positive="1")
conf.test_ov<-confusionMatrix(as.factor(pred.test_ov),(test_over$target),positive="1")
acc.train[1]<-conf.train_ov$overall['Accuracy']
sens.train[1]<-(conf.train_ov$byClass)[1]
spec.train[1]<-(conf.train_ov$byClass)[2]
acc.test[1]<-conf.test_ov$overall['Accuracy']
sens.test[1]<-(conf.test_ov$byClass)[1]
spec.test[1]<-(conf.test_ov$byClass)[2]
#Ploting ROC curve and AUC for test and train set
par(mfrow=c(1,2))
par(pty="s")
#For training set
roc_log.train_ov<-roc(train_over$target,pred.logit_train_ov,plot=T,col="#69b3a2",print.auc=T,legacy.axes=TRUE,percent = T,
                   xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Train Set")
#For test set
roc_log.test_ov<-roc(test_over$target,pred.logit_ov,plot=T,col="navyblue",print.auc=T,legacy.axes=TRUE,percent = T,
                  xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Test Set")

auc.train[1]=auc(train_over$target,model.logit_over$fitted.values)
auc.test[1]=auc(test_over$target,pred.logit_ov)




# B. LDA #############################################

#fitting a LDA model
model.lda_ov=lda(target~.,data=train_over,prior=c(prop.table(table(over$target))))

#making prediction for both train and test sets
pred.lda.test_ov=predict(model.lda_ov,test_over)
pred.lda.prob_ov=pred.lda.test_ov$posterior[,2]
pred.lda.train_ov=predict(model.lda_ov,train_over)


#calculate the error rate for both train and test sets
conf.train_ov<-confusionMatrix(data=pred.lda.train_ov$class,reference=(train_over$target),positive="1")
conf.test_ov<-confusionMatrix(data=pred.lda.test_ov$class,reference=(test_over$target),positive="1")
acc.train[2]<-conf.train_ov$overall['Accuracy']
sens.train[2]<-(conf.train_ov$byClass)[1]
spec.train[2]<-(conf.train_ov$byClass)[2]
acc.test[2]<-conf.test_ov$overall['Accuracy']
sens.test[2]<-(conf.test_ov$byClass)[1]
spec.test[2]<-(conf.test_ov$byClass)[2]


#Ploting ROC curve and AUC for test and train set
par(mfrow=c(1,2))
par(pty="s")
#For training set
roc_lda.train_ov<-roc(response=train_over$target,predictor=pred.lda.train_ov$posterior[,2],plot=T,col="#69b3a2",print.auc=T,legacy.axes=TRUE,percent = T,
                   xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Train Set")
#For test set
roc_lda.test_ov<-roc(response=test_over$target,predictor=pred.lda.prob_ov,plot=T,col="navyblue",print.auc=T,legacy.axes=TRUE,percent = T,
                  xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Test Set")

auc.train[2]=auc(train_over$target,pred.lda.train_ov$posterior[,2])
auc.test[2]=auc(test_over$target,pred.lda.prob_ov)



#C. K-NEAREST NEIGHBOUR ###############################################

standardized.X_ov=scale(over[,c(-2:-4,-6:-11,-24)])
quali.dummy_ov=dummy.data.frame(over[,c(2:4,6:11)])
#Merging the normalized data and one-hot encoded dummies
target<-over$target
data.knn_ov=cbind(standardized.X_ov,quali.dummy_ov,target)
train.knn_ov=data.knn_ov[ind,]
test.knn_ov=data.knn_ov[-ind,]

no_cores <- detectCores()-1  #Calculate the number of cores
cl <- makePSOCKcluster(no_cores)#create the cluster for caret to use
registerDoParallel(cl)

knn_fit_ov<-caret::train(target ~ .,data=train.knn_ov,method = 'knn',trControl =trainControl(method="repeatedcv",repeats = 3))
stopCluster(cl)
registerDoSEQ()               

#Prediction and calculating error rate on test set
knn.pred_ov=knn(train.knn_ov[,c(-83)],test.knn_ov[,c(-83)],train.knn_ov[,c(83)] ,k=knn_fit_ov$bestTune[[1]])#Best model in terms of accuracy is when k=9
conf.test_ov<-confusionMatrix(knn.pred_ov ,as.factor(test.knn_ov[,c(83)]),positive="1")

#Prediction and calculating error rate on training set
model.knn.train_ov=knn3(train.knn_ov[,c(-83)],as.factor(train.knn_ov[,c(83)]),k=knn_fit_ov$bestTune[[1]])
conf.train_ov<-confusionMatrix(predict(model.knn.train_ov,train.knn_ov[,c(-83)],type = "class"),as.factor(train.knn_ov[,83]),positive="1")

acc.train[3]<-conf.train_ov$overall['Accuracy']
sens.train[3]<-(conf.train_ov$byClass)[1]
spec.train[3]<-(conf.train_ov$byClass)[2]
acc.test[3]<-conf.test_ov$overall['Accuracy']
sens.test[3]<-(conf.test_ov$byClass)[1]
spec.test[3]<-(conf.test_ov$byClass)[2]
#Plotting ROC curve and AUC for test and train set
par(mfrow=c(1,2))
par(pty="s")
roc_knn.train_ov<-roc(train.knn_ov[,c(83)],predict(model.knn.train_ov,train.knn_ov[,c(-83)],type="prob")[,2],plot=T,col="#69b3a2",print.auc=T,legacy.axes=TRUE,
                   percent = T,xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Train Set")
roc_knn.test_ov<-roc(test.knn_ov[,c(83)],predict(model.knn.train_ov,test.knn_ov[,c(-83)],type="prob")[,2],plot=T,col="navyblue",print.auc=T,legacy.axes=TRUE,percent = T,
                  xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Test Set")

auc.train[3]=auc(train.knn_ov[,c(83)],predict(model.knn.train_ov,train.knn_ov[,c(-83)],type="prob")[,2])
auc.test[3]=auc(test.knn_ov[,c(83)],predict(model.knn.train_ov,test.knn_ov[,c(-83)],type="prob")[,2])


#D. RANDOM FOREST #############
target=recode_factor(target,'0'="no",'1'="yes")
data.rf_ov<-cbind(standardized.X_ov,over[,c(2:4,6:11)],target)
train.rf_ov<-data.rf_ov[ind,]
test.rf_ov<-data.rf_ov[-ind,]
 
no_cores <- detectCores()-1  #Calculate the number of cores
cl <- makePSOCKcluster(no_cores)#create the cluster for caret to use
registerDoParallel(cl)
#Tuning Hyperparameters for RF
param = trainControl(method = "cv",number = 5 , allowParallel = T,classProbs = T,summaryFunction = twoClassSummary)
RandomForest_Fit_ov = caret::train(target~.,train.rf_ov, method = 'rf',trControl = param, tuneGrid=data.frame(.mtry = seq(5,20, by=5)))

stopCluster(cl)
registerDoSEQ()

#Model Fitting based on the Grid Search
model.rf_ov=randomForest(target~., data = train.rf_ov, mtry =RandomForest_Fit_ov$bestTune[[1]],maxnodes=30,importance=T)

#Prediction and calculating error rate for test and training sets
#for test set
pred.rf_ov= predict(model.rf_ov,newdata = test.rf_ov[,-24],probability=T)
pred.rf.prob_ov= predict(model.rf_ov,newdata = test.rf_ov[,-24],type="prob")
conf.test_ov<-confusionMatrix(pred.rf_ov ,test.rf_ov$target,positive="yes")

#for training set
pred.rf.train_ov= predict(model.rf_ov,newdata = train.rf_ov[,-24],probability=T)
pred.rf.train.prob_ov=predict(model.rf_ov,newdata = train.rf_ov[,-24],type="prob")
conf.train_ov=confusionMatrix(pred.rf.train_ov,train.rf_ov$target, positive="yes")

acc.train[4]<-conf.train_ov$overall['Accuracy']
sens.train[4]<-(conf.train_ov$byClass)[1]
spec.train[4]<-(conf.train_ov$byClass)[2]
acc.test[4]<-conf.test_ov$overall['Accuracy']
sens.test[4]<-(conf.test_ov$byClass)[1]
spec.test[4]<-(conf.test_ov$byClass)[2]

#ROC curve and AUC value for both train and test set
par(mfrow=c(1,2))
par(pty="s")
roc_rf.train_ov<-roc(train.rf_ov$target,(pred.rf.train.prob_ov[,2]),plot=T,
                                         col="#69b3a2",print.auc=T,legacy.axes=TRUE,percent = T,xlab="False Positive percentage",
                                         ylab="True Positive percentage",lwd=5,main="Train Set")

roc_rf.test_ov<-roc(test.rf_ov$target,pred.rf.prob_ov[,2],plot=T,col="navyblue",print.auc=T,percent = T,
                                       xlab="False Positive percentage",ylab="True Positive percentage",lwd=5,main="Test Set")

auc.train[4] = auc(train.rf_ov$target,(pred.rf.train.prob_ov[,2]))

auc.test[4] = auc(test.rf_ov$target,pred.rf.prob_ov[,2])

#CLASSIFICATION EVALUATION ############################
classification.eval_balanced=data.frame(Model=c("Logistic","LDA","KNN","Random forest"),Accuracy_train=acc.train,
                                   Sensitivity_train=sens.train, Specifivity_train=spec.train, AUC_train=auc.train,
                                   Accuracy_test=acc.test, Sensitivity_test=sens.test, Specificity_test=spec.test, AUC_test=auc.test)
classification.eval_balanced

# plot ROC curves together in one single plot
rocs_train_bal<- function(){
  plot(roc_log.train_ov, col = "green", lty=1, main = "ROCs train set")
  lines(roc_lda.train_ov, col = "red", lty=2)
  lines(roc_knn.train_ov, col = "orange", lty=3)
  lines(roc_rf.train_ov, col = "blue", lty=4)
  legend(57,40,legend=c("Logistic", "LDA","KNN","Random Forest"),
         col=c("green", "red", "orange","blue"),lty=1:4, cex=0.55)
}
rocs_test_bal<- function(){
  plot(roc_log.test_ov, col = "green", lty=1, main = "ROCs test set")
  lines(roc_lda.test_ov, col = "red", lty=2)
  lines(roc_knn.test_ov, col = "orange", lty=3)
  lines(roc_rf.test_ov, col = "blue", lty=4)
  legend(57,40,legend=c("Logistic", "LDA","KNN","Random Forest"),
         col=c("green", "red", "orange","blue"),lty=1:4, cex=0.55)
}

par(mfrow=c(1,2))
#ROCS train set
rocs_train_bal()
#ROCs test set
rocs_test_bal()
