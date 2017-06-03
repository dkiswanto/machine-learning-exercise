# Time Counter Execution
start.time <- Sys.time()

# Import dataset
source("dataset/loadmnist.R")
source("dataset/visual.R")
# source("util.R")

# Notes : 
# mnistdata = list, index = 1,2, name $labels and $images
# mnistdata$labels = vektor integer
# mnistdata$images = class matrix nrows=size_data, ncol=28*28   
# mnistdata$images[1,] = class integer (vector) per image, length dimmension 785 (28*28)

# Load mnistdata
size_data <- 1000
mnistdata <- loadmnist(size_data)

##### Question 1.a
cat("## Question 1.a\n")
cat("100 First Image saved to Rplots.pdf page 1\n")
visual(mnistdata$images[1:100,])
cat("100 First Labels Rplots.pdf page 1 : \n")
print(mnistdata$labels[1:100])

##### Question 1.b.i
cat("\n## Question 1.b.i\n")
cat("Tidak seperti Matlab, di R-programming, untuk menghitung eigen diperlukan vector column (dimensinya kesamping)\n")
cat("Jadi tidak diperlukan transpose pada Matrix\n")

##### Question 1.b.ii
cat("\n## Question 1.b.ii\n")
cat("Substraksi Matrix dengan RowMeans\n")
dataset = mnistdata$images
mean_data = rowMeans(dataset)
dataset_zero_centered = dataset - mean_data

##### Question 1.b.iii
cat("\n## Question 1.b.iii\n")
cat("Menghitung Covariance\n")
cov_b = cov(dataset_zero_centered)
# print(cov_b)
nrow(cov_b)
ncol(cov_b)

##### Question 1.b.iv
cat("\n## Question 1.b.iv\n")
cat("Menghitung Vektor dan Nilai Eigen\n")
eig = eigen(cov_b)
eig_vec = eig$vectors
eig_val = eig$values
# cat("\nVektor Eigen /\ Vrow\n")
# print(eig_vec)
# cat("\nNilai Eigen /\ L \n")
# print(eig_val)

##### Question 1.b.v
cat("\n## Question 1.b.v\n")
cat("Berdasarkan dokumentasi help(eigien), hasil vektor dan nilai eigen sudah terurut secara descending\n")

##### Question 1.b.vi
cat("\n## Question 1.b.vi\n")
cat("Visualisasi hasil Vektor Eigen disimpan di Rplots.pdf page 2\n")
visual(t(eig_vec[,1:64]))

##### Question 1.b.vii
cat("\n## Question 1.b.vii\n")
cat("Mengubah Datset ke Dimensi PCA dengan 64 First Principle Componen\n")
pca = t(eig_vec[,1:64]) %*% t(dataset)

##### Question 1.b.viii
cat("\n## Question 1.b.vii\n")
cat("Dimensi Data setiap Object pada domain PCA adalah :", nrow(pca), "Dimensi\n")

##### Question 1.c
cat("\n## Question 1.c\n")
cat("Reconstruction PCA to Native Domain\n")
reverse_data = eig_vec[,1:64] %*% pca
cat("Visualisasi 100 Data Rekonstruksi di Rplots.pdf page 3\n")
visual(t(reverse_data)[1:100,])

##### Question 1.c
cat("\n## Question 1.d\n")
correct_construction = sum(eig_val[1:64])/sum(eig_val)
cat("Percentage of correct reconstruction :", correct_construction, "\n")

# Time Calculation
time.taken <- Sys.time() - start.time
cat("\nTime Exection : ", time.taken, "\n")
