
#Installing these packages took like.. four hours.


library(caret) # For data splitting and accuracy calculation
library(e1071) # For Naive Bayes
library(ggplot2) # For data visualization
library(dplyr)
library(corrplot)
library(rpart)
library(rpart.plot)
apples <- read.csv("apple_quality.csv")


# Modified our Data to add extra features to show correlations + neg correlations. Got this from the dplyr library at: # https://dplyr.tidyverse.org/reference/mutate.html

apples <- apples %>%
  mutate(
    SizeCrunchiness = Size * Crunchiness,      # New feature for positive correlation
    AcidJuice = Acidity * Juiciness ,    # Another feature for positive correlation
    SizeDSweet = Size / Sweetness, 
    RipeDSweet = Ripeness / Sweetness,
  ) 


summary(apples)
head(apples)

cor_matrix <- cor(apples %>% select_if(is.numeric), use = "complete.obs")

# Custom color palette from CGPT to go between the apple colors
col <- colorRampPalette(c("#ef5449", "#ffefe4", "#b29831"))(200)


corrplot(cor_matrix, method = "circle", type = "upper", order = "hclust",
         tl.col = "black", tl.srt = 25,
         addCoef.col = "black",  
         col = col,  
         title = "Correlation Matrix of Apples",  
         diag = FALSE)  
# The closer to 1 it is the more it is correlated, the closer to -1, the less it is. 

ggplot(apples, aes(x = as.factor(Quality), y = Sweetness, fill = as.factor(Quality))) + 
  geom_violin(trim = FALSE, alpha = 0.7) + 
  geom_jitter(width = 0.2, color = "black", size = 0.5, alpha = 0.6) +  # Add jittered points
  scale_fill_viridis_d() +  labs(title = "Apple Sweetness by Quality", x = "Quality", y = "Sweetness") +
  theme_bw() 

ggplot(apples, aes(x = Weight, fill = as.factor(Quality))) +
  geom_density(alpha = 0.7) +
  scale_fill_viridis_d() +
  labs(title = "Density Plot of Weight by Quality", x = "Weight", y = "Density") +
  theme_bw()

ggplot(apples, aes(x = Sweetness, y = Weight)) +
  geom_point(aes(color = as.factor(Quality)), alpha = 0.6) +
  facet_wrap(~Quality) +
  scale_color_viridis_d() +
  labs(title = "Sweetness vs. Size, Faceted by Quality", x = "Sweetness", y = "Size") +
  theme_classic()

#Density plot as suggested to us by CGPT, we followed the similiar template for the rest of the plots. 

set.seed(777)
#My favourite number <3

indexes <- createDataPartition(apples$Quality, p = 0.7, list = FALSE)
trainer <- apples[indexes, ]
tester <- apples[-indexes, ]



rmodel <- rpart(Quality ~ ., data = trainer,control = rpart.control(cp = 0.0038, minsplit = 60, minbucket = 15, maxdepth = 7))

print(rmodel)

predictions <- predict(rmodel, tester, type = "class") 
tester$Quality <- as.factor(tester$Quality)
predictions <- factor(predictions, levels = levels(tester$Quality))


conr <- confusionMatrix(predictions, tester$Quality)
print(conr)

# I got an accuracy rate of about 70%, which I thought was good, but not good enough yet. So we wanted to try K-Fold cross validation as shown by CGPT to get more accuracy. But we got a lower number, so instead we decided to add modifications to the rpart model instead. 


train_control <- trainControl(method = "cv", number = 10)
grid <- expand.grid(.cp = seq(0.01, 0.1, by = 0.01))
model <- train(Quality ~ ., data = trainer, method = "rpart",
               trControl = train_control,
               tuneGrid = grid)
print(model)

bayes <- naiveBayes(Quality ~ ., data = trainer, laplace =2, usekernel = TRUE)
bayespredict <- predict(bayes,tester)

bayespredict <- factor(bayespredict, levels = levels(tester$Quality))
conBayes <- caret::confusionMatrix(bayespredict, tester$Quality)

print(conBayes)

raccuracy <- conr$overall['Accuracy']
naccuracy <- conBayes$overall['Accuracy']


cat("Accuracy of rpart model:", raccuracy, "\n")
cat("Accuracy of Naïve Bayes model:", naccuracy, "\n")

rpart.plot(rmodel, type = 0, extra = 101) 
importance <- varImp(model)$importance
print(importance)

