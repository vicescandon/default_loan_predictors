#### Clean memory
rm(list = ls())

#### Load libraries
library(MASS)
library(glmnet)
library(car)
library(verification)       ## For ROC based analysis
library(caret)              ## For refined confusion matrix

#### Load functions
source("C:/Users/victo/OneDrive/Documents/UNIFI/Classes/DS4E/DS4E-Functions.R")

#### Original data
file <- "C:/Users/victo/OneDrive/Documents/UNIFI/Classes/DS4E/final project/Data-original.txt"
data <- read.table(file = file, header = TRUE, sep = "\t", na.strings = ".")
head(data, 3) 

#### Live data
file <- "C:/Users/victo/OneDrive/Documents/UNIFI/Classes/DS4E/final project/Data-live.txt"
data.live <- read.table(file = file, header = TRUE, sep = "\t", na.strings = ".")
head(data.live, 3)

# Convert LOAN STATUS to binary 0 and 1 
data$Current_loan_status <- ifelse(data$Current_loan_status == "DEFAULT", 1, 0)
data$Current_loan_status <- as.factor(data$Current_loan_status)


#### Adjust factors 
## data
data1 <- data
data1$home_ownership <- as.factor(data1$home_ownership)
data1$loan_intent <- as.factor(data1$loan_intent)
data1$loan_grade <- as.factor(data1$loan_grade)
data1$historical_default <- as.factor(data1$historical_default)
data1$Current_loan_status <- as.factor(data1$Current_loan_status)
data <- data1
## data.live
data1 <- data.live
data1$home_ownership <- as.factor(data1$home_ownership)
data1$loan_intent <- as.factor(data1$loan_intent)
data1$loan_grade <- as.factor(data1$loan_grade)
data1$historical_default <- as.factor(data1$historical_default)
data.live <- data1

######## PRELIMINARIES OF DATA #########
dim(data)         ## rows x cols
colnames(data)    ## Names of the columns
head(data, 3)     ## First 3 lines
str(data)         ## Structure
summary(data)     ## Summary

#Analytic primaries
#### Dependent variable
#### Extract the variable
x1 <- data$Current_loan_status
#### Some stats
tab <- table(x1, exclude = NULL)  ## Absolute frequencies
tab

tabp <- 100 * tab/sum(tab)        ## Percentage frequencies
tabp

#### Plot
par(mfrow = c(1,1))
barplot(tabp, main = "Loan status % frequencies")

# Ajustar los parámetros gráficos
par(mfrow = c(7, 1), mar = c(3, 3, 2, 1), oma = c(0, 0, 2, 0), cex = 0.7)

# Boxplot for Customer Income vs. Current Loan Status
boxplot(customer_income ~ Current_loan_status, data = data, horizontal = TRUE,
        main = "Income vs Loan Status", 
        xlab = "Income", ylab = "Loan Status")

# Boxplot for Employment Duration vs. Current Loan Status
boxplot(employment_duration ~ Current_loan_status, data = data, horizontal = TRUE,
        main = "Employment Duration vs Loan Status", 
        xlab = "Employment Duration", ylab = "Loan Status")

# Boxplot for Loan Amount vs. Current Loan Status
boxplot(loan_amnt ~ Current_loan_status, data = data, horizontal = TRUE,
        main = "Loan Amount vs Loan Status", 
        xlab = "Loan Amount", ylab = "Loan Status", ylim=c(0,100000))

# Boxplot for Interest Rate vs. Current Loan Status
boxplot(loan_int_rate ~ Current_loan_status, data = data, horizontal = TRUE,
        main = "Interest Rate vs Loan Status", 
        xlab = "Interest Rate", ylab = "Loan Status")

# Boxplot for Loan Term (Years) vs. Current Loan Status
boxplot(term_years ~ Current_loan_status, data = data, horizontal = TRUE,
        main = "Loan Term vs Loan Status", 
        xlab = "Loan Term", ylab = "Loan Status")

# Boxplot for Credit History Length vs. Current Loan Status
boxplot(cred_hist_length ~ Current_loan_status, data = data, horizontal = TRUE,
        main = "Credit History Length vs Loan Status", 
        xlab = "Credit History Length", ylab = "Loan Status")

#categoriral variables
ind <- c("home_ownership", "Current_loan_status")
table(data[, ind])
prop.table(table(data[, ind]), margin = 1)

ind <- c("loan_intent", "Current_loan_status")
table(data[, ind])
prop.table(table(data[, ind]), margin = 1)

ind <- c("loan_grade", "Current_loan_status")
table(data[, ind])
prop.table(table(data[, ind]), margin = 1)

ind <- c("historical_default", "Current_loan_status")
table(data[, ind])
prop.table(table(data[, ind]), margin = 1)


############## LEGISTIC FULL MODEL ################
#train/test split
#### Set random numbers seed (to avoid changes each time it is run)
set.seed(1234)
#### Number of obs in the test set (20% of NROW(data))
n.test <- 0.20 * NROW(data)
#### Which rows to include in the test set
ind <- sample(x = 1:NROW(data), size = n.test)
#### Split
data.orig <- data                         ## Store original data
data.test <- data[ind, , drop = FALSE]    ## Store train data
data.train <- data[-ind, , drop = FALSE]  ## Store test data
##### Check the sizes
cat("NROW(original)", NROW(data), "NROW(train)", NROW(data.train),
    "NROW(test)", NROW(data.test))


#LOGISTIC FULL MODEL
#### Fit
fit <- glm(formula = Current_loan_status ~ customer_age + customer_income + home_ownership + employment_duration +
             loan_intent + loan_grade + loan_amnt + loan_int_rate + term_years + historical_default + cred_hist_length,
           data = data.train, family = binomial(link = "logit"))
R2 <- (fit$null.deviance - fit$deviance) / fit$null.deviance
#### Copy for future use
fit.full <- fit
form.full <- formula(fit.full)
#### Summary
summary(fit.full)

#model performance
y <- data.test$Current_loan_status
Mfull<-.fit.stats.bin(fit = fit.full, y = y, x = data.test)

#_____________________________________________
#### REGULARIZED REGRESSION LASSO
alpha<-1
#### Dep. var
y <- data.train$Current_loan_status
#### Indep. vars
xmat <- .glmnet.xmat(formula = formula(fit.full), data = data.train)
####
lambda.all <- exp(seq(from = -10, to = 7, by = .2))

#### Which one is the best lambda?
cvfit <- cv.glmnet(x = xmat, y = y, alpha = alpha, lambda = lambda.all,
                   family = "binomial", type.measure = "deviance", nfolds = 20)
par( mfrow = c(1,1) )
plot(cvfit)

#### Print best lambdas
cat("min(lambda) = ", cvfit$lambda.min, "1se(lambda) = ", cvfit$lambda.1se, "\n")

#### Refit using the best lambda
lambda <- cvfit$lambda.1se
fit.lasso <- glmnet(x = xmat, y = y, family = "binomial", alpha = alpha,
              lambda = lambda, standardize = TRUE)

#model performance
y <- data.test$Current_loan_status
xmat <- model.matrix(object = form.full, data = data.test)[, -1, drop = FALSE]
Mlasso<-.fit.stats.bin(fit = fit.lasso, y = y, x = xmat)
#___________________________________
#______________________________________
library(ranger)             ## For Random Forests
library(car)                ## For recode
library(verification)       ## For ROC based analysis
library(caret)              ## For refined confusion matrix
#Random Forest
formula<-form.full
feat <- .features(x = formula)

#fit
fit.rf <- ranger(formula = formula, data = data.train, 
              probability = 2000, splitrule = "gini", importance = "impurity", 
              num.trees = 750, mtry = 4,
              min.node.size = 5, min.bucket = 1, max.depth = NULL,
              keep.inbag = TRUE, oob.error = TRUE, verbose = TRUE, seed = 2000)
.rf.errorMeasures(fit = fit.rf, data = data.train)
#### Check the 'type' error measure as function of num.trees
par(mfrow = c(2,1), mar = c(4,4,1,0.2))
type <- "me"
error.cum <- .rf.errorPlot(fit = fit.rf, data = data.train, type = type,
                           what = c("all"))
error.cum <- .rf.errorPlot(fit = fit.rf, data = data.train, type = type,
                           what = c("oob"))
par(mfrow = c(1,1), mar = c(4,4,1,0.2))
error.cum <- .rf.errorPlot(fit = fit.rf, data = data.train, type = type,
                           what = c("oob"))

#____________________________
#variable importance
#### Variable importance
par(mar = c(4, 9, 0.5, 0.5))
vimp <- .rf.varImp(fit = fit.rf, plot = TRUE)

#____________________________
#MODEL PERFORMANCE
#### Redefines quantities for safety reasons
data <- data.test
y <- data$Current_loan_status
#### Compute
Mrf <- .fit.stats.bin(fit = fit.rf, y = y, x = data)
Mrf

#_________________________
#GRADIENT BOOSTING
#### Load functions
source("C:/Users/victo/OneDrive/Documents/UNIFI/Classes/DS4E/DS4E-Functions.R")

#variables fixing for R not to crash
#### Original data
file <- "C:/Users/victo/OneDrive/Documents/UNIFI/Classes/DS4E/final project/Data-original.txt"
data <- read.table(file = file, header = TRUE, sep = "\t", na.strings = ".")
head(data, 3) 

#### Live data
file <- "C:/Users/victo/OneDrive/Documents/UNIFI/Classes/DS4E/final project/Data-live.txt"
data.live <- read.table(file = file, header = TRUE, sep = "\t", na.strings = ".")
head(data.live, 3)

# Convert LOAN STATUS to binary 0 and 1 
data$Current_loan_status <- ifelse(data$Current_loan_status == "DEFAULT", 1, 0)
data$Current_loan_status <- as.factor(data$Current_loan_status)

## data
data1 <- data
data1$home_ownership <- as.factor(data1$home_ownership)
data1$loan_intent <- as.factor(data1$loan_intent)
data1$loan_grade <- as.factor(data1$loan_grade)
data1$historical_default <- as.factor(data1$historical_default)
data1$Current_loan_status <- as.factor(data1$Current_loan_status)
data <- data1
## data.live
data1 <- data.live
data1$home_ownership <- as.factor(data1$home_ownership)
data1$loan_intent <- as.factor(data1$loan_intent)
data1$loan_grade <- as.factor(data1$loan_grade)
data1$historical_default <- as.factor(data1$historical_default)
data.live <- data1

#### Transform all character vars in factor (otherwise we may get errors in gbm())
data <- .character.2.factor(data = data, exclude = NULL, include = NULL)
data.live <- .character.2.factor(data = data.live, exclude = NULL, include = NULL)

#independent to numeric
data$Current_loan_status<-as.numeric(data$Current_loan_status)
#verify numeric
class(data$Current_loan_status)

#### Set random numbers seed (to avoid changes each time it is run)
set.seed(1234)
#### Number of obs in the test set (20% of NROW(data))
n.test <- 0.20 * NROW(data)
#### Which rows to include in the test set
ind <- sample(x = 1:NROW(data), size = n.test)
#### Split
data.orig <- data                         ## Store original data
data.test <- data[ind, , drop = FALSE]    ## Store train data
data.train <- data[-ind, , drop = FALSE]  ## Store test data
#### Check the sizes
cat("NROW(original)", NROW(data), "NROW(train)", NROW(data.train),
    "NROW(test)", NROW(data.test))
#formula
formula<-Current_loan_status ~ customer_age + customer_income + home_ownership + employment_duration +
                     loan_intent + loan_grade + loan_amnt + loan_int_rate + term_years + historical_default + cred_hist_length
#FIT GRADIENT BOOSTING
library(gbm)
library(xgboost)
set.seed(1)
fit.gb <- gbm(formula = formula, data = data.train, distribution = "bernoulli",
           var.monotone = NULL, train.fraction = 0.8, cv.folds = 10,
           interaction.depth = 3, n.minobsinnode = 10, n.trees = 2000, shrinkage = 0.05,
           bag.fraction = 0.5,
           keep.data = TRUE, verbose = FALSE)
#_____________________________
#PREDICTIONS FULL

#CUTOFF
#### Train data
data <- data.train
#### Proportion of 1 in data (as a reference)
y <- as.numeric(as.character(data$Current_loan_status))
prop1 <- mean(y)
pred.p <- predict(object = fit.full, newdata = data, type = "response")
## quantile rule
coff.full <- quantile(x = fitted(fit.full), prob = 1 - prop1)
class <- .classify(x = pred.p, coff = coff.full)

#confussion matrix
cmat <- table(class = class, true = y, exclude = NULL)
cmat
prop.table(cmat, margin = 1)
cmat.s <- confusionMatrix(
  data = as.factor(class), reference = as.factor(y),
  positive = "1")
cmat.s

#_________________________________
#PREDICTION LASSO
#CUTOFF
#### Train data
data <- data.train
#### Proportion of 1 in data (as a reference)
y <- as.numeric(as.character(data$Current_loan_status))
prop1 <- mean(y)
xmat <- .glmnet.xmat(formula = formula(fit.full), data = data)
pred.lasso <- predict(object = fit.lasso, newx = xmat, type = "response")
## quantile rule
coff.lasso <- quantile(x = pred.lasso, prob = 1 - prop1)
class <- .classify(x = pred.lasso, coff = coff.lasso)

#confussion matrix
cmat <- table(class = class, true = y, exclude = NULL)
cmat
prop.table(cmat, margin = 1)
cmat.s <- confusionMatrix(
  data = as.factor(class), reference = as.factor(y),
  positive = "1")
cmat.s

#PREDICTION FOREST TREE
#CUTOFF
#### Train data
data <- data.train
#### Proportion of 1 in data (as a reference)
y <- as.numeric(as.character(data$Current_loan_status))
prop1 <- mean(y)
pred.rf <- predict(object = fit.rf, data = data, type = "response")$predictions[, "1"]
## quantile rule
coff.rf <- quantile(x = pred.rf, prob = 1 - prop1)
class <- .classify(x = pred.rf, coff = coff.rf)

#confussion matrix
cmat <- table(class = class, true = y, exclude = NULL)
cmat
prop.table(cmat, margin = 1)
cmat.s <- confusionMatrix(
  data = as.factor(class), reference = as.factor(y),
  positive = "1")
cmat.s

#______________________
#all cutoff of models
cbind(coff.full,coff.lasso,coff.rf)

#_______________________
#PREDICTIONS ON LIVE DATA
#### Predict
#### Data
data <- data.orig
#### I use lasso because it is the most complex
fit.rf <- ranger(formula = formula, data = data, 
                 probability = 2000, splitrule = "gini", importance = "impurity", 
                 num.trees = 750, mtry = 4,
                 min.node.size = 5, min.bucket = 1, max.depth = NULL,
                 keep.inbag = TRUE, oob.error = TRUE, verbose = TRUE, seed = 2000)

#### Predict
pred.rf <- predict(object = fit.rf, data = data.live, type = "response")$predictions[,"1"]
y <- as.numeric(as.character(data$Current_loan_status))
prop1 <- mean(y)
coff.rf <- quantile(x = pred.rf, prob = 1 - prop1)
class <- .classify(x = pred.rf, coff = coff.rf)
pred <- data.frame(data.live, pred = pred.rf, class = class, check.names = FALSE)
#### Write on a file
file <- "C:/Users/victo/OneDrive/Documents/UNIFI/Classes/DS4E/final project/Data-live-predictions.txt"
write.table(x = pred, file = file, quote = FALSE, sep = "\t",
  na = ".", row.names = FALSE, col.names = TRUE)
