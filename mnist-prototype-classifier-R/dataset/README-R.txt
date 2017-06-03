
Example use of MNIST data in R

First, source all the .R files in the directory:

> source("sourcedir.R")
> sourceDir("./")

Next, load the MNIST data

> mnistdata <- loadmnist()

Note: If you do not have enough memory to read in all the 60,000
images, you can read a smaller amount, for example 5,000 as follows

> N <- 5000
> mnistdata <- loadmnist( N )

Show the first 100 digits

> visual(mnistdata$X[1:100,])

Show the corresponding labels

> print(mnistdata$y[1:100])

