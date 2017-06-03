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
size_data <- 1000
mnistdata <- loadmnist(size_data)

# cat("Substraksi Matrix dengan RowMeans\n")
dataset = mnistdata$images
mean_data = rowMeans(dataset)
dataset_zero_centered = dataset - mean_data

# cat("Menghitung Covariance\n")
cov_b = cov(dataset_zero_centered)

# cat("Menghitung Vektor dan Nilai Eigen\n")
eig = eigen(cov_b)
eig_vec = eig$vectors
eig_val = eig$values

##### Question 1.e
cat("## Question 1.e\n")
data_precentage = c()
for(i in 1:784){
    correct_construction = sum(eig_val[1:i])/sum(eig_val)
    data_precentage = c(data_precentage, correct_construction)
}
plot(data_precentage, type = "o", col = "red", 
    xlab = "K First Principle Component", ylab = "Precentage Reconstruction",
    main = "Overview Precentage Correction")
cat("Data Saved in Rplots.pdf page 1\n")

##### Question 1.f
cat("## Question 1.f\n")
misc_data = c()
check_dim = 200
for(dim in seq(from=5, to=check_dim, by=5)){
    pca = t(eig_vec[,1:dim]) %*% t(dataset)
    pca = t(pca)
    misc = 0
    for(i in 1:nrow(pca)){
        true_label = mnistdata$labels[i]
        prediction = knn_based_classifier(pca[i,], pca, mnistdata$labels)
        # cat("True Label", true_label, ",Prediction", prediction, "\n")
        if(true_label != prediction){
            misc = misc + 1
        } 
        # break
    }
    misc = misc / nrow(pca) * 100
    cat("Miscalculation with dim:", dim , "=", misc, "\n")
    misc_data = c(misc_data, misc)
    # break
}

plot(seq(from=5, to=check_dim, by=5), type="l", misc_data,
    xlab = "K First Principle Component", ylab = "Miscalculation Rete (%)",
    main = "Overview Miscalculation")

# Time Calculation
time.taken <- Sys.time() - start.time
cat("\nTime Exection : ", time.taken, "\n")
