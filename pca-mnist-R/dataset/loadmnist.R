loadmnist <- function( N=60000 )
{
  # Loads the MNIST data from the files:
  #
  # train-images-idx3-ubyte
  # train-labels-idx1-ubyte
  #
  # ...and puts the images in X and the labels in y in a list
  # returned to the calling function.
  #
  # The input argument N is the number of digits to read. It
  # defaults to the maximum (60000)

  data <- list()

  # Make sure we don't try to read more than there actually is
  if (N>60000) N <- 60000
  
  # Read in the data
  data$images <- readBin('dataset/train-images-idx3-ubyte', 'int', n=16+28*28*N,
                                                size=1, signed=FALSE)
  data$images <- data$images[17:length(data$images)];
  dim(data$images) <- c(28*28, N)
  data$images <- t(data$images)
        
  # Read in the labels
  data$labels <- readBin('dataset/train-labels-idx1-ubyte', 'int', n=8+N,
                                             size=1, signed=FALSE)
  data$labels <- data$labels[9:length(data$labels)]

  # Return the data
  data

}
