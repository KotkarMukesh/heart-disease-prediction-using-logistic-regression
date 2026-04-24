
library(dplyr)
library(tidyr)
library(readr)
library(caret)
library(ggplot2)
install.packages('gplots')
library(gplots)
library(GGally)

#importing data 
df = read.csv("C:/Users/lenovo/Downloads/heart.csv")
head(df)



#dimension of the dataset
dim(df)

#getting the summary of the dataset
summary(df)

#getting information about the dataframe
str(df)



# Exploratory data analysis 
# performing basic EDA on heart data 

#count missing values in each column
colSums(is.na(df))


#visualizing the distribution of the target variable 
ggplot(df, aes(x = target)) + geom_bar(fill = 'skyblue', color = 'purple', stat = "count") + labs(title = "Distribution of Target Variable", x = "Target", y = "Frequency")

#now we will plot a histogram to show the relation between heart diseases and and age
ggplot(df, aes(x= age, fill = factor(target))) + geom_histogram(binwidth = 4, position = "dodge", color = "grey") + facet_wrap(~target, scales = "free_y") + scale_fill_discrete(labels = c("No Disease", "disease"))



# next we will try to find a relation between gender and heart disease 
ggplot(df, aes(x = factor(sex), fill = factor(target))) +
  geom_bar(position = "dodge") +
  labs(
    x = "Gender (0 = Female, 1 = Male)",
    y = "Frequency",
    fill = "Disease Status"
  ) +
  scale_fill_manual(
    values = c("0" = "blue", "1" = "red"),
    labels = c("No disease", "Disease")
  )


#correlation matrix 
ggcorr(df, label = TRUE, label_size = 2.5, hjust = 1, layout.exp = 2)



#Exploratory Data Analysis reveals that certain factors like gender, type of chest pain, fasting blood sugar level, etc. are categorical 
#variables and are not suitable for model training. We will encode these variables into "factor" data type in R for organized and improved 
#understanding of this information. This encoding allows for efficient handling of categorical variables in statistical models and data analysis.

heart = df %>% 
  mutate(sex = as.factor(sex),
         cp = as.factor(cp), 
         fbs = as.factor(restecg),
         exang = as.factor(exang),
         slope = as.factor(slope),
         ca = as.factor(ca),
         thal = as.factor(ca),
         target = as.factor(target)
         )
str(heart)



# normalization and data splitting 
"""at first we will select the features with higher relation with target
variable. then perform data normalizaion for stable and faster training of
logistic regression model, after that we split the data into 80% for training 
and 20% data is for prediction purpose 
"""


#feature selection
features = df[, c('age', 'sex',  'cp', 'trestbps', 'chol', 'restecg', 'thalach', 
                   'exang', 'oldpeak', 'slope', 'ca', 'thal')]
target = df$target

preprocessParams = preProcess(features, method = c("center", "scale"))
features_normalized = predict(preprocessParams, features)



#splitting the data
split = createDataPartition(target, p = 0.8, list = FALSE)
X_train = features_normalized[split, ]
X_test = features_normalized[-split, ]
Y_train = target[split]
Y_test = target[-split]

#shape of the training and test sets
print(paste("X_train shape:", paste(dim(X_train), collapse = 'x')))
print(paste("X_test shape:", paste(dim(X_test), collapse = "x")))

'''
Now that we have our data normalized and split into train and test sets,
we are ready to train the Logistic Regression model on this data.
'''

train_data = as.data.frame(cbind(target = Y_train, X_train))

#training logistic regression model
model = glm(target ~ ., data= train_data, family = "binomial")





#making orediction on the test set

prediction = predict(model, newdata = as.data.frame(X_test), type = "response")


binary_prediciton = ifelse(prediction >= 0.5, 1, 0)

result = data.frame(actual = Y_test, predicted = binary_prediciton)



confusionMatrix(data = as.factor(binary_prediciton), reference = as.factor(Y_test), positive = "1")



#create a confusion matrix
conf_matrix = table(factor(binary_prediciton, levels = c("0", "1")), factor(Y_test, levels = c("0", "1")))

#setting dimensions
dimnames(conf_matrix) = list(Actual = c("0","1"), predictes = c("0","1"))

#plot the fourthfold plot with color main title
fourfoldplot(conf_matrix, color = c("blue", "red"), main = "Confusion Matrix")







#new values of the model
test_data = as.data.frame(cbind(target = Y_test, X_test))

# Making predictions on the test set
predictions = predict(model, newdata = as.data.frame(test_data[, -1]),type ="response")

# Converting probabilities to binary predictions based on threshold 0.5
binary_predictions = ifelse(predictions >= 0.5, 1, 0)

# Combining actual values and predicted values into a data frame
result = data.frame(actual = test_data$target, predicted = binary_predictions)

# Displaying the results
print(result)






