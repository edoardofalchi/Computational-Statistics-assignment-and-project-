set.seed(321)

library(ggplot2) #for a nice plotting


#------------------------Defining functions------------------------------------------------------------

#--------------The data generating process (DGP)-------------------------------------------------------

# This function generates an artificial sample with sample size N from the specified data generating 
# process. beta is the vector with the coefficients. 

data_generator <- function(N, beta){
  ##
  X   <- cbind(rep(1, N), rnorm(N, 0, sqrt(1.5)))
  ##
  eps  <- rnorm(N, 0, sqrt(10))
  Y    <- X %*% beta + eps
  data <- data.frame("Y"=Y, "X"=X[,2])
  ##
  return(data)
}


#---------------Mean Squared Error Calculator----------------------------------------------------------


#The mean squared error (MSE) is the mean of the squared difference between the fitted values and the
#actual realisations for our dependent variable y. The difference itself is called the residual. 
#When we do a linear regression with the function lm, R stores the residuals. We can access them 
#with $residuals. Then we have to square them and take the mean. 


mse <- function(lm){
  mean(lm$residuals^2)
}


#----------------Average prediction error Calculator----------------------------------------------------------

# The average prediction error tells us, how good the predictive power of our fitted model is, when
# we use it for new data from the same data generating process. The data which we are use for fitting  
# the model is the training-sample, and we test the model with our test sample. We take the realizations
# of the independent variable X and calculate the corresponding predictions for the y-values 
# with our fitted model. Then we compare them with the actual realizations of the dependent variable:
# We take the mean of the squared difference. 
 
# The function takes a test-data-set, the number of observations, N,
# and the estimated coefficients of our model as an input.   


a_pred_error <- function(data_test, beta_hat, N){
  
  X <- as.matrix(cbind(rep(1, N), data_test[-1]))
  
  # In the line above, we are generating the X-matrix. Firstly, we need a column for the constant. 
  # Then we are adding the test-data to the matrix, (without the dependent variable). 
  
  y_pred <-  X %*% beta_hat
  mean((data_test[,1] - y_pred)^2)
  
}

#--------------------Fitting and evaluate a linear model-----------------------------------------------

#-------------------Generate a test- and a training sample---------------------------------------------

N <- 1000
beta_true <- c(5, -0.5) #true parameters, not observed in reality

training_sample <- data_generator(N = N, beta = beta_true)
test_sample <- data_generator(N = N, beta = beta_true) 




#--------------Fitting a linear model with the training sample----------------------------------------- 

lm_training <- lm(Y ~ X, data = training_sample) 
beta_hat <- coef(lm_training) #storing the estimates for beta
print(beta_hat)



#----Calculating the MSE and the average prediction error with our defined function---------------------

MSE <- mse(lm = lm_training)
print(MSE)

pred_error <- a_pred_error(data_test = test_sample, beta_hat = beta_hat, N = N)
print(pred_error)



#-------------------------------Overfitting the model----------------------------------------------------

# We know, that the explanatory variable x enters only linear in our data generating process.
# But in reality, we don't know for sure, how the structure of the model in the background
# looks like. Now we want to see, what happens when we take the data from our linear model 
# and estimate polynomial models with different degrees. 

#-------------------------------Generating new variables--------------------------------------------------

# When we want to estimate a polynomial with our linear regression, we have to generate new variables:
# We have to take the X-variable, take it to the power of 2,3,... and store the results. In the end we 
# want to calculate MSE and average prediction errors. Therefore, we have to transform the X-variables
# from both the training- and the test-sample. 

X_power_training <- matrix(NaN,N,5) #We want polynomials with degrees up to 5
X_power_test <- matrix(NaN,N,5)

# We take X (second column of the sample) and take it to the power of r. Then we store it. 


for (r in 1:5){
  
  X_power_training[,r] <- training_sample[,2]^r 
  X_power_test[,r] <- test_sample[,2]^r
  
}

# We store the new generated variables together with the realizations of Y in a data frame to 
# fit a linear model and calculate errors (again for both samples). 


data_power_training <- data.frame("Y" = training_sample[,1], X_power_training)
data_power_test <- data.frame("Y" = test_sample[,1], X_power_test)


#----------------------Fitting polynomial models with different degrees------------------------


# We are running 6 linear regression with polynomials up to degree 5, calculating the MSE
# and the average prediction error for each result. 

#Generating vectors to store the results

MSE_result <- c(NaN)
pred_error_result <- c(NaN)

# We are considering the special case with just the constant (degree 0) outside of the loop

l_model <- lm(Y ~ 1, data = data_power_training)

MSE_result[1] <- mse(lm = l_model) 
pred_error_result[1] <- mean((data_power_test[,1] - coef(l_model))^2)


for (r in 1:5){ # r is the maximal degree of the polynomials
  
  l_model <- lm(data_power_training[,1] ~ X_power_training[,1:r])
  MSE_result[r+1] <- mse(lm = l_model)
  pred_error_result[r+1] <- a_pred_error(data_test = data_power_test[,1:(r+1)],
                                         beta_hat = coef(l_model),
                                         N = N)
}  

print(MSE_result)
print(pred_error_result)



#----------------------------Simulation study-------------------------------------------------------------

# In our simulation above, we only considered one realization of our training- and test-sample. 
# Now we are using a simulation study to assess, how the MSE and the average prediction error behave
# on average, when we increase the degrees of the polynomial. We are repeating our simulation 
# 1000 times, store the results and calculate averages.  

num_sim <- 1000  #number of repetitions
result_MSE <- data.frame(c(rep(NaN,6))) #data frames for our results
result_pred <- data.frame(c(rep(NaN,6)))


for (r in 1:num_sim){
  set.seed(100 + r)

  training_sample <- data_generator(N = N, beta = beta_true) #Drawing test and training sample
  test_sample <- data_generator(N = N, beta = beta_true)
  
  #---------------------------------------------------------------
  
  X_power_training <- matrix(NaN,N,5) #Generating new variables
  X_power_test <- matrix(NaN,N,5)
  
  for (i in 1:5){
    X_power_training[,i] <- training_sample[,2]^i 
    X_power_test[,i] <- test_sample[,2]^i
  }
  
  data_power_training <- data.frame("Y" = training_sample[,1], X_power_training) 
  data_power_test <- data.frame("Y"=test_sample[,1], X_power_test)
  
  #---------------------------------------------------------------
  
  #Fitting models for polynomials with different degrees
  
  MSE_result <- c(NaN)
  pred_error_result <- c(NaN)
  
  l_model <- lm(Y ~ 1, data = data_power_training) #special case
  
  MSE_result[1] <- mse(lm = l_model) 
  pred_error_result[1] <- mean((data_power_test[,1] - coef(l_model))^2)
  
    
  for (k in 1:5){ 
    
    l_model <- lm(data_power_training[,1] ~ X_power_training[,1:k])
    MSE_result[k+1] <- mse(lm = l_model) #k+1 due to constant case
    pred_error_result[k+1] <- a_pred_error(data_test = data_power_test[,1:(k+1)], #k+1 due Y-variable
                                           beta_hat = coef(l_model),
                                           N = N)
  }  
  
  result_MSE[r] <- MSE_result #storing the results
  result_pred[r] <- pred_error_result
  
}

#-----------------------Calculating averages------------------------------------------------------


#Each column of the result data frames represents the findings from one step of the loop. We take 
#the mean of the rows, to get the average result for the different polynomials. 

MSE_mean_sim <- rowMeans(result_MSE)
pred_mean_sim <- rowMeans(result_pred)

print(MSE_mean_sim)
print(pred_mean_sim)


#-------------------------Plotting the results-----------------------------------------------------


data = data.frame(c(0:5),MSE_mean_sim,pred_mean_sim)

ggplot(data = data, aes()) + 
  geom_point(mapping = aes(x = data[,1] , y = MSE_mean_sim, color = "red" )) +
  geom_point(mapping = aes(x = data[,1],y = pred_mean_sim, color = "blue")) +
  ggtitle("Average MSE and prediction error") + 
  theme(plot.title = element_text(hjust = 0.5)) +
  xlab("Degree of polynomial") + ylab("MSE, APE") +
  scale_color_discrete(name="", labels = c("APE", "MSE"))
  

#-------------------------Interpretation----------------------------------------------------------

#Generally speaking, we do not really care how well the method works on the training data. Rather, we are interested
#in the accuracy of the predictions that we obtain when we apply our method to previously unseen test data. Ultimately, we want
# to have test_MSE i.e. the APE as low as possible.

#The training MSE declines monotonically as flexibility increases, while the APE has an hump-shaped pattern:
#As with the training MSE, the test MSE initially declines as the level of flexibility increases. However,
#at some point the test MSE levels off and then starts to increase again. 
#All this, indeed, is reflected in our simulation study plot
# We saw in our simulation study, that on average the MSE decreases and the APE increases, when we 
# increase the degree of the polynomial (after the constant case). When you increase the degree, the
# fit of the model to the training-data-set gets better and better.As model flexibility increases, training MSE will decrease,
#but the test MSE may not. So having a small training MSE but a large test MSE, is a signal of overfitting the data. ////////////////////This is so, because the OLS method 
# minimizes the sum of the squared residuals, which is N*MSE. When we increase the number of parameters, 
# the MSE is at least as high (then the coefficient of this new parameter is equal to zero), 
# or the sum is smaller than before. 

# So the MSE is always decreasing, but we don't care so much about the MSE, because to get the smallest
# MSE we would just set the predicted y-values equal to the observed ones, leading to an MSE equal to zero. 
# Instead we are interested, how good the model is in predicting values, not used to fit the model. 
# This is captured in the average prediction error. When we increase the degree of the polynomials, then 
# the average prediction error gets bigger and bigger, leading (on average) to less and less accurate 
# predictions, because we overfitted the model. 

# Therefore, we can conclude, that the MSE is a measure for how close the fit of the model is to the 
# training data, but can't be used to assess the accuracy of the predictions from this model.



# Our data generating process fulfills the all four assumptions from the lecture. Therefore, 
# the OLS-estimator is unbiased estimator. This means,that (on average), our estimation-results are close to the theoretical (true) parameters
#  In this setting, the OLS estimator is also a consistent estimator. When we would increase
# the sample size, the accuracy of our prediction would (on average) get better and better, due to 
# the Central Limit Theorem. With N=1000 we have already a quite large sample.
#Therefore, increasing sample size or the number of MC simulations would help us to show that 
#simulation results match theoretical estimator properties.
#This explains, why the estimated coefficients above were close to the coefficients from the data generating process.
#DGP comes from Normal, if we had a differnt DGP probably we would need to apply a different method to get low prediction errors

print(coef(lm_training))
print(beta_true)

# An other observation is, that the MSE is roughly equal to 10, which is also the value of the variance 
# of the error term in our data generating process. This is not a coincidence, because with this setup,
# the MSE is an asymptotically unbiased estimator of the error-variance. 





#----------------------------Reference---------------------------------------------------------------------

# James, G., D. Witten, T. Hastie, and R. Tibshirani (2013): An introduction
# to statistical learning, vol. 112, Springer. 






