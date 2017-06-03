# Time Counter Execution
start.time <- Sys.time()

# Import dataset
source("dataset/loadmnist.R")
source("dataset/visual.R")
source("util.R")

# Notes : 
# mnistdata = list, index = 1,2, name $labels and $images
# mnistdata$labels = vektor integer
# mnistdata$images = class matrix nrows=size_data, ncol=28*28   
# mnistdata$images[1,] = class integer (vector) per image, length dimmension 785 (28*28)

# Load mnistdata
size_data <- 5000
mnistdata <- loadmnist(size_data)

##### Question 7.a
cat("## Question 7.a\n")
random_n = 100
random_vector <- sample.int(size_data, random_n)
temp_images <- matrix(NA, nrow=random_n, ncol=28*28)
temp_labels <- c()
counter = 1
for(i in random_vector){
    temp_labels <- c(temp_labels, mnistdata$labels[i])
    temp_images[counter,] <- mnistdata$images[i,]
    counter <- counter + 1
}
cat("Image Saved to Rplots.pdf page 1\n")
visual(temp_images)
cat("Labels Rplots.pdf page 1 : \n")
print(temp_labels)

##### Question 7.b
cat("\n## Question 7.b\n")
n_divide <- 2500
training_image <- mnistdata$images[1:n_divide,]
training_label <- head(mnistdata$labels, n_divide)
testing_image <- mnistdata$images[(n_divide + 1): size_data, ]
testing_label <- tail(mnistdata$labels, n_divide)

cat("training_label_length : ", length(training_label), '\n')
cat("testing_label_length : ", length(testing_label), '\n')
cat("Mean Image / Prototype Image Classifier saved to Rplots.pdf page 2\n")

label_count <- c(0,0,0,0,0,0,0,0,0,0)
class_image_protoype = matrix(0, nrow=10, ncol=28*28)

index <- 1
for(label in training_label){
    if(label == 0){
        label_count[1] <- label_count[1] + 1
        class_image_protoype[1,] <- class_image_protoype[1,] + training_image[index,]

    } else if (label == 1){
        label_count[2] <- label_count[2] + 1
        class_image_protoype[2,] <- class_image_protoype[2,] + training_image[index,]

    } else if (label == 2){
        label_count[3] <- label_count[3] + 1
        class_image_protoype[3,] <- class_image_protoype[3,] + training_image[index,]

    } else if (label == 3){
        label_count[4] <- label_count[4] + 1
        class_image_protoype[4,] <- class_image_protoype[4,] + training_image[index,]

    } else if (label == 4){
        label_count[5] <- label_count[5] + 1
        class_image_protoype[5,] <- class_image_protoype[5,] + training_image[index,]

    } else if (label == 5){
        label_count[6] <- label_count[6] + 1
        class_image_protoype[6,] <- class_image_protoype[6,] + training_image[index,]

    } else if (label == 6){
        label_count[7] <- label_count[7] + 1
        class_image_protoype[7,] <- class_image_protoype[7,] + training_image[index,]

    } else if (label == 7){
        label_count[8] <- label_count[8] + 1
        class_image_protoype[8,] <- class_image_protoype[8,] + training_image[index,]

    } else if (label == 8){
        label_count[9] <- label_count[9] + 1
        class_image_protoype[9,] <- class_image_protoype[9,] + training_image[index,]

    } else if (label == 9){
        label_count[10] <- label_count[10] + 1
        class_image_protoype[10,] <- class_image_protoype[10,] + training_image[index,]
    }

    index <- index + 1
}

for(i in 1:10){
    class_image_protoype[i,] <- class_image_protoype[i,] / label_count[i]
}
visual(class_image_protoype)

##### Question 7.c Prototype classifer
cat("\n## Question 7.c\n")
actual_row <- c("Act0", "Act1", "Act2", "Act3", "Act4", "Act5", "Act6", "Act7", "Act8", "Act9")
predicted_col <- c("Pred0", "Pred1", "Pred2", "Pred3", "Pred4", "Pred5", "Pred6", "Pred7", "Pred8", "Pred9")
confusion_matrix_prototype <- matrix(0, nrow=10, ncol=10 , dimnames = list(actual_row, predicted_col))
cat("Processing Prototype Classifier....\n" )
for(i in 1:2500){
    prediction_label <- prototype_based_classifier(testing_image[i,], class_image_protoype)
    
    actual_label <- testing_label[i]
    # print(testing_image[i,])
    # print(actual_label)

    confusion_matrix_prototype[actual_label + 1, prediction_label + 1] <- confusion_matrix_prototype[actual_label + 1, prediction_label + 1] + 1
    # cat("Process KNN", i ,": actual ", actual_label, " prediction ", prediction_label, "\n" )
}
cat("\nConfusion Matrix Prototype Classifier : \n") 
print(confusion_matrix_prototype); cat("\n")
get_accuracy(confusion_matrix_prototype)

    
##### Question 7.d Nearest neighbor classifer
cat("\n## Question 7.d\n")
confusion_matrix_knn <- matrix(0, nrow=10, ncol=10 , dimnames = list(actual_row, predicted_col))
cat("Processing KNN Classifier....\n" )
for(i in 1:2500){
    prediction_label <- knn_based_classifier(testing_image[i,], training_image, training_label)
    actual_label <- testing_label[i]
    confusion_matrix_knn[actual_label + 1, prediction_label + 1] <- confusion_matrix_knn[actual_label + 1, prediction_label + 1] + 1
    # cat("Process KNN", i ,": actual ", actual_label, " prediction ", prediction_label, "\n" )
}
cat("\nConfusion Matrix KNN Classifier : \n")
print(confusion_matrix_knn); cat("\n")
get_accuracy(confusion_matrix_knn)


# Time Calculation
time.taken <- Sys.time() - start.time
cat("\n\nTime Exection : ", time.taken, "\n")
