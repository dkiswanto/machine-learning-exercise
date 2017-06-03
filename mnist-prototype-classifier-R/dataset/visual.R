#
# visual - display a set of small images
#
# A        the images as row vectors
# cols     [optional] number of columns in display
# bgblack  [optional] if this flag is set the background is black
# border   [optional] border pixels (default is 0)
#

visual <- function( A, cols=0, bgblack=0, border=0 )
{

  if (cols==0) cols <- round(sqrt(dim(A)[1]))
  
  # Transpose to have each column be an image
  A <- t(A)

  # Get maximum absolute value (it represents white or black; zero is gray)
  maxi <- max(max(abs(A)))
  mini <- -maxi

  # This is the side of the window
  tdim <- sqrt(dim(A)[1])

  # Helpful quantities
  tdimm <- tdim-1
  tdimp <- tdim+1
  rows <- ceiling(dim(A)[2]/cols)
  
  # Initialization of the image
  if (bgblack) { bgval <- mini } else { bgval <- maxi }
  I <- matrix(bgval,tdim*rows+rows-1+(2*border),tdim*cols+cols-1+(2*border))

  for (i in 0:(rows-1)) {
    for (j in 0:(cols-1)) {
    
      if (i*cols+j+1>dim(A)[2]) {
        # This leaves it at background color
      }
      else
      {
        # This sets the patch
        patch <- A[,i*cols+j+1]
        dim(patch) <- c(tdim,tdim)
        I[(border+i*tdimp+1):(border+i*tdimp+tdim), 
	  (border+j*tdimp+1):(border+j*tdimp+tdim)] <- t(patch)
      }
    
    }
  }

  I <- t(I)

  nx <- dim(I)[1]
  ny <- dim(I)[2]
  
  image(I[,ny:1], col=gray(0:256/256), axes = FALSE)
  
}
