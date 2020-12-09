library(arules)

# read dataset
linearSet <- read.csv('./linear.csv')
linearSet <- subset(linearSet, select=-c(X, param_kernel, params, 
                                         split0_test_neg_mean_absolute_error, split1_test_neg_mean_absolute_error, split2_test_neg_mean_absolute_error, split3_test_neg_mean_absolute_error, split4_test_neg_mean_absolute_error, rank_test_neg_mean_absolute_error, 
                                         split0_train_neg_mean_absolute_error, split1_train_neg_mean_absolute_error, split2_train_neg_mean_absolute_error, split3_train_neg_mean_absolute_error, split4_train_neg_mean_absolute_error,
                                         split0_train_neg_mean_squared_error, split1_train_neg_mean_squared_error, split2_train_neg_mean_squared_error, split3_train_neg_mean_squared_error, split4_train_neg_mean_squared_error, rank_test_neg_mean_squared_error,
                                         split0_test_neg_mean_squared_error, split1_test_neg_mean_squared_error, split2_test_neg_mean_squared_error, split3_test_neg_mean_squared_error, split4_test_neg_mean_squared_error,
                                         split0_test_r2, split1_test_r2, split2_test_r2, split3_test_r2, split4_test_r2, rank_test_r2,
                                         split0_train_r2, split1_train_r2, split2_train_r2, split3_train_r2, split4_train_r2))
summary(linearSet)

# discretize values
linearSet$mean_fit_time <- discretize(linearSet$mean_fit_time, breaks=3, labels=c("small", "medium", "big"))
linearSet$std_fit_time <- discretize(linearSet$std_fit_time, breaks=3, labels=c("small", "medium", "big"))
linearSet$mean_score_time <- discretize(linearSet$mean_score_time, breaks=3, labels=c("small", "medium", "big"))
linearSet$std_score_time <- discretize(linearSet$std_score_time, breaks=3, labels=c("small", "medium", "big"))

linearSet$mean_test_neg_mean_absolute_error <- discretize(linearSet$mean_test_neg_mean_absolute_error, breaks=3, labels=c("small", "medium", "big"))
linearSet$std_test_neg_mean_absolute_error <- discretize(linearSet$std_test_neg_mean_absolute_error, breaks=3, labels=c("small", "medium", "big"))
linearSet$mean_train_neg_mean_absolute_error <- discretize(linearSet$mean_train_neg_mean_absolute_error, breaks=3, labels=c("small", "medium", "big"))
linearSet$std_train_neg_mean_absolute_error <- discretize(linearSet$std_train_neg_mean_absolute_error, breaks=3, labels=c("small", "medium", "big"))

linearSet$mean_test_neg_mean_squared_error <- discretize(linearSet$mean_test_neg_mean_squared_error, breaks=3, labels=c("small", "medium", "big"))
linearSet$std_test_neg_mean_squared_error <- discretize(linearSet$std_test_neg_mean_squared_error, breaks=3, labels=c("small", "medium", "big"))
linearSet$mean_train_neg_mean_squared_error <- discretize(linearSet$mean_train_neg_mean_squared_error, breaks=3, labels=c("small", "medium", "big"))
linearSet$std_train_neg_mean_squared_error <- discretize(linearSet$std_train_neg_mean_squared_error, breaks=3, labels=c("small", "medium", "big"))

linearSet$mean_test_r2 <- discretize(linearSet$mean_test_r2, breaks=3, labels=c("small", "medium", "big"))
linearSet$std_test_r2 <- discretize(linearSet$std_test_r2, breaks=3, labels=c("small", "medium", "big"))
linearSet$mean_train_r2 <- discretize(linearSet$mean_train_r2, breaks=3, labels=c("small", "medium", "big"))
linearSet$std_train_r2 <- discretize(linearSet$std_train_r2, breaks=3, labels=c("small", "medium", "big"))

linearSet$param_C <- discretize(linearSet$param_C, breaks=2, labels=c("small", "big"))
linearSet$param_epsilon <- discretize(linearSet$param_epsilon, breaks=2, labels=c("small", "big"))

# convert to transcations
linearTR <- as(linearSet, "transactions")

# generate rules with apriori algorithm
aParam  = new("APparameter", "confidence"=0.6, "support"=0.05, "minlen"=1, "maxlen"=8)
aParam@target ="rules"

linearRules <-apriori(linearTR, aParam)
summary(linearRules)

linearParamRules <- subset(linearRules, subset = lhs %pin% "param_epsilon=" & lhs %pin% "param_C=" & size(lhs) == 2)
summary(linearParamRules)
write.csv(as(linearParamRules, "data.frame"), "./rules/linear-rules.csv", row.names=TRUE)

######################

# read dataset
rbfSet <- read.csv('./rbf.csv')
rbfSet <- subset(rbfSet, select=-c(X, param_kernel, params, 
                                         split0_test_neg_mean_absolute_error, split1_test_neg_mean_absolute_error, split2_test_neg_mean_absolute_error, split3_test_neg_mean_absolute_error, split4_test_neg_mean_absolute_error, rank_test_neg_mean_absolute_error, 
                                         split0_train_neg_mean_absolute_error, split1_train_neg_mean_absolute_error, split2_train_neg_mean_absolute_error, split3_train_neg_mean_absolute_error, split4_train_neg_mean_absolute_error,
                                         split0_train_neg_mean_squared_error, split1_train_neg_mean_squared_error, split2_train_neg_mean_squared_error, split3_train_neg_mean_squared_error, split4_train_neg_mean_squared_error, rank_test_neg_mean_squared_error,
                                         split0_test_neg_mean_squared_error, split1_test_neg_mean_squared_error, split2_test_neg_mean_squared_error, split3_test_neg_mean_squared_error, split4_test_neg_mean_squared_error,
                                         split0_test_r2, split1_test_r2, split2_test_r2, split3_test_r2, split4_test_r2, rank_test_r2,
                                         split0_train_r2, split1_train_r2, split2_train_r2, split3_train_r2, split4_train_r2))
summary(rbfSet)

# discretize values
rbfSet$mean_fit_time <- discretize(rbfSet$mean_fit_time, breaks=3, labels=c("small", "medium", "big"))
rbfSet$std_fit_time <- discretize(rbfSet$std_fit_time, breaks=3, labels=c("small", "medium", "big"))
rbfSet$mean_score_time <- discretize(rbfSet$mean_score_time, breaks=3, labels=c("small", "medium", "big"))
rbfSet$std_score_time <- discretize(rbfSet$std_score_time, breaks=3, labels=c("small", "medium", "big"))

rbfSet$mean_test_neg_mean_absolute_error <- discretize(rbfSet$mean_test_neg_mean_absolute_error, breaks=3, labels=c("small", "medium", "big"))
rbfSet$std_test_neg_mean_absolute_error <- discretize(rbfSet$std_test_neg_mean_absolute_error, breaks=3, labels=c("small", "medium", "big"))
rbfSet$mean_train_neg_mean_absolute_error <- discretize(rbfSet$mean_train_neg_mean_absolute_error, breaks=3, labels=c("small", "medium", "big"))
rbfSet$std_train_neg_mean_absolute_error <- discretize(rbfSet$std_train_neg_mean_absolute_error, breaks=3, labels=c("small", "medium", "big"))

rbfSet$mean_test_neg_mean_squared_error <- discretize(rbfSet$mean_test_neg_mean_squared_error, breaks=3, labels=c("small", "medium", "big"))
rbfSet$std_test_neg_mean_squared_error <- discretize(rbfSet$std_test_neg_mean_squared_error, breaks=3, labels=c("small", "medium", "big"))
rbfSet$mean_train_neg_mean_squared_error <- discretize(rbfSet$mean_train_neg_mean_squared_error, breaks=3, labels=c("small", "medium", "big"))
rbfSet$std_train_neg_mean_squared_error <- discretize(rbfSet$std_train_neg_mean_squared_error, breaks=3, labels=c("small", "medium", "big"))

rbfSet$mean_test_r2 <- discretize(rbfSet$mean_test_r2, breaks=3, labels=c("small", "medium", "big"))
rbfSet$std_test_r2 <- discretize(rbfSet$std_test_r2, breaks=3, labels=c("small", "medium", "big"))
rbfSet$mean_train_r2 <- discretize(rbfSet$mean_train_r2, breaks=3, labels=c("small", "medium", "big"))
rbfSet$std_train_r2 <- discretize(rbfSet$std_train_r2, breaks=3, labels=c("small", "medium", "big"))

rbfSet$param_C <- discretize(rbfSet$param_C, breaks=2, labels=c("small", "big"))
rbfSet$param_epsilon <- discretize(rbfSet$param_epsilon, breaks=2, labels=c("small", "big"))
rbfSet$param_gamma <- discretize(rbfSet$param_gamma, breaks=2, labels=c("small", "big"))

# convert to transcations
rbfTR <- as(rbfSet, "transactions")

# generate rules with apriori algorithm
aParam  = new("APparameter", "confidence"=0.6, "support"=0.05, "minlen"=1, "maxlen"=8)
aParam@target ="rules"

rbfRules <-apriori(rbfTR, aParam)
summary(rbfRules)

rbfParamRules <- subset(rbfRules, subset = lhs %pin% "param_epsilon=" & lhs %pin% "param_C=" & lhs %pin% "param_gamma=" & size(lhs) == 3)
summary(rbfParamRules)
write.csv(as(rbfParamRules, "data.frame"), "./rules/rbf-rules.csv", row.names=TRUE)

######################

# read dataset
polySet <- read.csv('./poly.csv')
polySet <- subset(polySet, select=-c(X, param_kernel, params, 
                                   split0_test_neg_mean_absolute_error, split1_test_neg_mean_absolute_error, split2_test_neg_mean_absolute_error, split3_test_neg_mean_absolute_error, split4_test_neg_mean_absolute_error, rank_test_neg_mean_absolute_error, 
                                   split0_train_neg_mean_absolute_error, split1_train_neg_mean_absolute_error, split2_train_neg_mean_absolute_error, split3_train_neg_mean_absolute_error, split4_train_neg_mean_absolute_error,
                                   split0_train_neg_mean_squared_error, split1_train_neg_mean_squared_error, split2_train_neg_mean_squared_error, split3_train_neg_mean_squared_error, split4_train_neg_mean_squared_error, rank_test_neg_mean_squared_error,
                                   split0_test_neg_mean_squared_error, split1_test_neg_mean_squared_error, split2_test_neg_mean_squared_error, split3_test_neg_mean_squared_error, split4_test_neg_mean_squared_error,
                                   split0_test_r2, split1_test_r2, split2_test_r2, split3_test_r2, split4_test_r2, rank_test_r2,
                                   split0_train_r2, split1_train_r2, split2_train_r2, split3_train_r2, split4_train_r2))
summary(polySet)

# discretize values
polySet$mean_fit_time <- discretize(polySet$mean_fit_time, breaks=3, labels=c("small", "medium", "big"))
polySet$std_fit_time <- discretize(polySet$std_fit_time, breaks=3, labels=c("small", "medium", "big"))
polySet$mean_score_time <- discretize(polySet$mean_score_time, breaks=3, labels=c("small", "medium", "big"))
polySet$std_score_time <- discretize(polySet$std_score_time, breaks=3, labels=c("small", "medium", "big"))

polySet$mean_test_neg_mean_absolute_error <- discretize(polySet$mean_test_neg_mean_absolute_error, breaks=3, labels=c("small", "medium", "big"))
polySet$std_test_neg_mean_absolute_error <- discretize(polySet$std_test_neg_mean_absolute_error, breaks=3, labels=c("small", "medium", "big"))
polySet$mean_train_neg_mean_absolute_error <- discretize(polySet$mean_train_neg_mean_absolute_error, breaks=3, labels=c("small", "medium", "big"))
polySet$std_train_neg_mean_absolute_error <- discretize(polySet$std_train_neg_mean_absolute_error, breaks=3, labels=c("small", "medium", "big"))

polySet$mean_test_neg_mean_squared_error <- discretize(polySet$mean_test_neg_mean_squared_error, breaks=3, labels=c("small", "medium", "big"))
polySet$std_test_neg_mean_squared_error <- discretize(polySet$std_test_neg_mean_squared_error, breaks=3, labels=c("small", "medium", "big"))
polySet$mean_train_neg_mean_squared_error <- discretize(polySet$mean_train_neg_mean_squared_error, breaks=3, labels=c("small", "medium", "big"))
polySet$std_train_neg_mean_squared_error <- discretize(polySet$std_train_neg_mean_squared_error, breaks=3, labels=c("small", "medium", "big"))

polySet$mean_test_r2 <- discretize(polySet$mean_test_r2, breaks=3, labels=c("small", "medium", "big"))
polySet$std_test_r2 <- discretize(polySet$std_test_r2, breaks=3, labels=c("small", "medium", "big"))
polySet$mean_train_r2 <- discretize(polySet$mean_train_r2, breaks=3, labels=c("small", "medium", "big"))
polySet$std_train_r2 <- discretize(polySet$std_train_r2, breaks=3, labels=c("small", "medium", "big"))

polySet$param_C <- discretize(polySet$param_C, breaks=2, labels=c("small", "big"))
polySet$param_epsilon <- discretize(polySet$param_epsilon, breaks=2, labels=c("small", "big"))
polySet$param_coef0 <- discretize(polySet$param_coef0, breaks=2, labels=c("small", "big"))
polySet$param_degree <- discretize(polySet$param_degree, breaks=2, labels=c("small", "big"))

# convert to transcations
polyTR <- as(polySet, "transactions")

# generate rules with apriori algorithm
aParam  = new("APparameter", "confidence"=0.6, "support"=0.05, "minlen"=1, "maxlen"=8)
aParam@target ="rules"

polyRules <-apriori(polyTR, aParam)
summary(polyRules)

polyParamRules <- subset(polyRules, subset = lhs %pin% "param_epsilon=" & lhs %pin% "param_C=" & lhs %pin% "param_coef0=" & lhs %pin% "param_degree=" & size(lhs) == 4)
summary(polyParamRules)
write.csv(as(polyParamRules, "data.frame"), "./rules/poly-rules.csv", row.names=TRUE)

######################

# read dataset
sigmoidSet <- read.csv('./sigmoid.csv')
sigmoidSet <- subset(sigmoidSet, select=-c(X, param_kernel, params, 
                                     split0_test_neg_mean_absolute_error, split1_test_neg_mean_absolute_error, split2_test_neg_mean_absolute_error, split3_test_neg_mean_absolute_error, split4_test_neg_mean_absolute_error, rank_test_neg_mean_absolute_error, 
                                     split0_train_neg_mean_absolute_error, split1_train_neg_mean_absolute_error, split2_train_neg_mean_absolute_error, split3_train_neg_mean_absolute_error, split4_train_neg_mean_absolute_error,
                                     split0_train_neg_mean_squared_error, split1_train_neg_mean_squared_error, split2_train_neg_mean_squared_error, split3_train_neg_mean_squared_error, split4_train_neg_mean_squared_error, rank_test_neg_mean_squared_error,
                                     split0_test_neg_mean_squared_error, split1_test_neg_mean_squared_error, split2_test_neg_mean_squared_error, split3_test_neg_mean_squared_error, split4_test_neg_mean_squared_error,
                                     split0_test_r2, split1_test_r2, split2_test_r2, split3_test_r2, split4_test_r2, rank_test_r2,
                                     split0_train_r2, split1_train_r2, split2_train_r2, split3_train_r2, split4_train_r2))
summary(sigmoidSet)

# discretize values
sigmoidSet$mean_fit_time <- discretize(sigmoidSet$mean_fit_time, breaks=3, labels=c("small", "medium", "big"))
sigmoidSet$std_fit_time <- discretize(sigmoidSet$std_fit_time, breaks=3, labels=c("small", "medium", "big"))
sigmoidSet$mean_score_time <- discretize(sigmoidSet$mean_score_time, breaks=3, labels=c("small", "medium", "big"))
sigmoidSet$std_score_time <- discretize(sigmoidSet$std_score_time, breaks=3, labels=c("small", "medium", "big"))

sigmoidSet$mean_test_neg_mean_absolute_error <- discretize(sigmoidSet$mean_test_neg_mean_absolute_error, breaks=3, labels=c("small", "medium", "big"))
sigmoidSet$std_test_neg_mean_absolute_error <- discretize(sigmoidSet$std_test_neg_mean_absolute_error, breaks=3, labels=c("small", "medium", "big"))
sigmoidSet$mean_train_neg_mean_absolute_error <- discretize(sigmoidSet$mean_train_neg_mean_absolute_error, breaks=3, labels=c("small", "medium", "big"))
sigmoidSet$std_train_neg_mean_absolute_error <- discretize(sigmoidSet$std_train_neg_mean_absolute_error, breaks=3, labels=c("small", "medium", "big"))

sigmoidSet$mean_test_neg_mean_squared_error <- discretize(sigmoidSet$mean_test_neg_mean_squared_error, breaks=3, labels=c("small", "medium", "big"))
sigmoidSet$std_test_neg_mean_squared_error <- discretize(sigmoidSet$std_test_neg_mean_squared_error, breaks=3, labels=c("small", "medium", "big"))
sigmoidSet$mean_train_neg_mean_squared_error <- discretize(sigmoidSet$mean_train_neg_mean_squared_error, breaks=3, labels=c("small", "medium", "big"))
sigmoidSet$std_train_neg_mean_squared_error <- discretize(sigmoidSet$std_train_neg_mean_squared_error, breaks=3, labels=c("small", "medium", "big"))

sigmoidSet$mean_test_r2 <- discretize(sigmoidSet$mean_test_r2, breaks=3, labels=c("small", "medium", "big"))
sigmoidSet$std_test_r2 <- discretize(sigmoidSet$std_test_r2, breaks=3, labels=c("small", "medium", "big"))
sigmoidSet$mean_train_r2 <- discretize(sigmoidSet$mean_train_r2, breaks=3, labels=c("small", "medium", "big"))
sigmoidSet$std_train_r2 <- discretize(sigmoidSet$std_train_r2, breaks=3, labels=c("small", "medium", "big"))

sigmoidSet$param_C <- discretize(sigmoidSet$param_C, breaks=2, labels=c("small", "big"))
sigmoidSet$param_epsilon <- discretize(sigmoidSet$param_epsilon, breaks=2, labels=c("small", "big"))
sigmoidSet$param_coef0 <- discretize(sigmoidSet$param_coef0, breaks=2, labels=c("small", "big"))

# convert to transcations
sigmoidTR <- as(sigmoidSet, "transactions")

# generate rules with apriori algorithm
aParam  = new("APparameter", "confidence"=0.6, "support"=0.05, "minlen"=1, "maxlen"=8)
aParam@target ="rules"

sigmoidRules <-apriori(sigmoidTR, aParam)
summary(sigmoidRules)

sigmoidParamRules <- subset(sigmoidRules, subset = lhs %pin% "param_epsilon=" & lhs %pin% "param_C=" & lhs %pin% "param_coef0=" & size(lhs) == 3)
summary(sigmoidParamRules)
write.csv(as(sigmoidParamRules, "data.frame"), "./rules/sigmoid-rules.csv", row.names=TRUE)
