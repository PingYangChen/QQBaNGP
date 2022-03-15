library(pracma)
# 3-D Hartmann function
Hartmann3D <- function(x) {
  xx <- repmat(x, 4, 1)
  al <- c(1., 1.2, 3., 3.2)
  am <- rbind(c(3., 10., 30.), 
              c(.1, 10., 35.), 
              c(3., 10., 30.), 
              c(.1, 10., 35.))
  pm <- 1e-4*rbind(c(3689., 1170., 2673.), 
                   c(4699., 4387., 7470.), 
                   c(1091., 8732., 5547.), 
                   c( 381., 5743., 8828.))
  val = -sum(al*exp(-rowSums(am*((xx - pm)^(2.)))))
  return(val)
}

bntest2 <- function(x, w, v) {
  if (w == 0) {
    val <- Hartmann3D(x) - 2*(v^2)*(1-v)
  }
  else if (w == 1) {
    val <- Hartmann3D(x)
  }
  else {
    stop(sprintf('w = %d not defined', w))
  }
  return(val)
}

Griewank <- function(x) {
  # transform: [0, 1] -> [-5, 5]
  xx <- x*10 - 5.
  d <- length(x)
  ap <- 0; mp <- 1
  for (i in 1:d) {
    ap <- ap + xx[i]^2/4000.
    mp <- mp*cos(xx[i]/sqrt(i))
  }
  val <- ap - mp + 1
  return(val)
  # Global min = 0 at xx = (0, 0, ..., 0)
}

Matyas2D <- function(x) {
  # transform: [0, 1] -> [-5, 5]
  xx <- x*10 - 5.
  val <- 0.26*sum(xx*xx) - 0.48*prod(xx)
  return(val)
}

Himmelblau2D <- function(x) {
  # transform: [0, 1] -> [-5, 5]
  xx <- x*10 - 5.
  val <- (xx[1]*xx[1] + xx[2] - 11)^2 + (xx[1] + xx[2]*xx[2] - 7)^2
  return(val/100)
  # Global min = 0 at 
  # xx = (3.0, 2.0), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)
}

Sphere <- function(x) {
  # transform: [0, 1] -> [-5, 5]
  xx <- x*10 - 5.
  val <- sum(xx^2)
  return(val)
}

GramacyLee1D <- function(x) {
  # transform: [0, 1] -> [0.5, 2.5]
  xx <- x*2 + .5
  val <- sin(10*pi*xx)/(2*xx) + (xx - 1)^4
  return(val)
}

linear1D <- function(x) {
  val <- -.5*(x - .5)
  return(val)
}

beta1D <- function(x) {
  val <- -5*(x^2)*(1 - x) + .2
  return(val)
}


x1v1_Sphere1D_0linear <- function(x, z, w, v) {
  if (w == 0) {
    val <- Sphere(x) + linear1D(v)
  }
  else if (w == 1) {
    val <- Sphere(x)
  }
  else {
    stop(sprintf('w = %d not defined', w))
  }
  return(val)
}

x1v1_Sphere1D_0beta <- function(x, z, w, v) {
  if (w == 0) {
    val <- Sphere(x) + beta1D(v)
  }
  else if (w == 1) {
    val <- Sphere(x)
  }
  else {
    stop(sprintf('w = %d not defined', w))
  }
  return(val)
}

x1v1_GramacyLee1D_0linear <- function(x, z, w, v) {
  if (w == 0) {
    val <- GramacyLee1D(x) + linear1D(v)
  }
  else if (w == 1) {
    val <- GramacyLee1D(x)
  }
  else {
    stop(sprintf('w = %d not defined', w))
  }
  return(val)
}


x1v1_GramacyLee1D_0beta <- function(x, z, w, v) {
  if (w == 0) {
    val <- GramacyLee1D(x) + beta1D(v)
  } 
  else if (w == 1) {
    val <- GramacyLee1D(x)
  }
  else {
    stop(sprintf('w = %d not defined', w))
  }
  return(val)
}

x1v2_GramacyLee1D_0himmelblau <- function(x, z, w, v) {
  if (w == 0) {
    val <- GramacyLee1D(x) + Himmelblau2D(v)
  }
  else if (w == 1) {
    val <- GramacyLee1D(x)
  }
  else {
    stop(sprintf('w = %d not defined', w))
  }
  return(val)
}

x2v1_Matyas2D_0linear <- function(x, z, w, v) {
  if (w == 0) {
    val <- Matyas2D(x) + linear1D(v)
  }
  else if (w == 1) {
    val <- Matyas2D(x)
  }
  else {
    stop(sprintf('w = %d not defined', w))
  }
  return(val)
}

x2v1_Matyas2D_0beta <- function(x, z, w, v) {
  if (w == 0) {
    val <- Matyas2D(x) + beta1D(v)
  }
  else if (w == 1) {
    val <- Matyas2D(x)
  }
  else {
    stop(sprintf('w = %d not defined', w))
  }
  return(val)
}

x2v1_Griewank_0linear <- function(x, z, w, v) {
  if (w == 0) {
    val <- Griewank(x) + linear1D(v)
  }
  else if (w == 1) {
    val <- Griewank(x)
  }
  else {
    stop(sprintf('w = %d not defined', w))
  }
  return(val)
}

x2v1_Griewank_0beta <- function(x, z, w, v) {
  if (w == 0) {
    val <- Griewank(x) + beta1D(v)
  }
  else if (w == 1) {
    val <- Griewank(x)
  }
  else {
    stop(sprintf('w = %d not defined', w))
  }
  return(val)
}


x2v2_Griewank_0himmelblau <- function(x, z, w, v) {
  if (w == 0) {
    val <- Griewank(x) + Himmelblau2D(v)
  }
  else if (w == 1) {
    val <- Griewank(x)
  }
  else {
    stop(sprintf('w = %d not defined', w))
  }
  return(val)
}




x3v1_Hartmann3D_0linear <- function(x, z, w, v) {
  if (w == 0) {
    val <- Hartmann3D(x) + linear1D(v)
  }
  else if (w == 1) {
    val <- Hartmann3D(x)
  }
  else {
    stop(sprintf('w = %d not defined', w))
  }
  return(val)
}

x3v1_Hartmann3D_0beta <- function(x, z, w, v) {
  if (w == 0) {
    val <- Hartmann3D(x) + beta1D(v)
  }
  else if (w == 1) {
    val <- Hartmann3D(x)
  }
  else {
    stop(sprintf('w = %d not defined', w))
  }
  return(val)
}

x3v1_Hartmann3D_0GramacyLee1D <- function(x, z, w, v) {
  if (w == 0) {
    val <- Hartmann3D(x) + GramacyLee1D(v)
  }
  else if (w == 1) {
    val <- Hartmann3D(x)
  }
  else {
    stop(sprintf('w = %d not defined', w))
  }
  return(val)
}

x3v2_Hartmann3D_0himmelblau <- function(x, z, w, v) {
  if (w == 0) {
    val <- Hartmann3D(x) + Himmelblau2D(v)
  }
  else if (w == 1) {
    val <- Hartmann3D(x)
  }
  else {
    stop(sprintf('w = %d not defined', w))
  }
  return(val)
}


x2z1w1v2_ver00 <- function(x, z, w, v) {
  
  zAdd <- sum(z)
  if (w == 0) {
    val <- Matyas2D(x) + Sphere(v[1]) - 5
  }
  else if (w == 1) {
    val <- Matyas2D(x) + Sphere(v) - 15
  }
  else if (w == 2) {
    val <- Matyas2D(x) + Sphere(v[1]) + 5
  }
  else if (w == 3) {
    val <- Matyas2D(x) + Sphere(v)
  }  
  else {
    stop(sprintf('w = %d not defined', w))
  }
  return(val + zAdd)
  
}

x2z1w1v2_ver01 <- function(x, z, w, v) {
  
  zAdd <- sum(z)
  if (w == 0) {
    val <- Matyas2D(x) + Sphere(v[1]) - 5
  }
  else if (w == 1) {
    val <- Matyas2D(x) + Himmelblau2D(v) - 15
  }
  else if (w == 2) {
    val <- Matyas2D(x) + Sphere(v[1]) + 5
  }
  else if (w == 3) {
    val <- Matyas2D(x) + Himmelblau2D(v)
  }  
  else {
    stop(sprintf('w = %d not defined', w))
  }
  return(val + zAdd)
  
}

# 
# franke <- function(x) {
#   x1 <- x[1]
#   x2 <- x[2]
#   term1 <-  0.75*exp( -3*(x1 - 2/9)^2/4  - (9*x2 - 2)^2/4 )
#   term2 <-  0.75*exp( -(9*x1 + 1)^2/49 - (9*x2 + 1)/10  )
#   term3 <-  0.50*exp( -(9*x1 - 7)^2/4  - (9*x2 - 3)^2/4 )
#   term4 <- -0.20*exp( -(9*x1 - 4)^2    - (9*x2 - 7)^2   )
#   y <- term1 + term2 + term3 + term4
#   return(y)
# }
# ###
# library(plotly)
# ng <- 100
# x1 <- x2 <- seq(0, 1, length = ng)
# ymat <- matrix(0, ng, ng)
# for (i in 1:ng) {
#   for (j in 1:ng) {
#     ymat[i,j] <- franke(c(x1[i], x2[j]))
#   }
# }
# fig <- plot_ly(
#   type = 'surface',
#   contours = list(
#     x = list(show = TRUE, start = 0, end = 1, size = 0.05, color = 'white'),
#     z = list(show = TRUE, start = 0, end = 1, size = 0.05)),
#   x = ~x1,
#   y = ~x2,
#   z = ~ymat)


###

# 
x2z1w1v2_ver02 <- function(x, z, w, v) {

  zz <- exp(((z[1]+1) - 2.5)/1.25)

  x1 <- x[1]
  x2 <- x[2]
  
  if (w == 0) {
    vv <- (c(v[1], v[1]) + 1)*10*zz
  }
  else if (w == 1) {
    vv <- .8 + v*3*zz
  }
  else if (w == 2) {
    vv <- (c(v[1], v[1]) + 1)*6*zz 
  }
  else if (w == 3) {
    vv <- (v + 1)*5*zz
  }
  else {
    stop(sprintf('w = %d not defined', w))
  }
  term1 <-  0.75*exp( (-(9*x1 - 2)^2/(4*vv[1])  - (9*x2 - 2)^2/(4*vv[2])  )*zz^2 )*sqrt(zz)
  term2 <-  0.75*exp( (-(9*x1 + 1)^2/(49*vv[1]) - (9*x2 + 1)^1/(10*vv[2]) )*zz^2 )*sqrt(zz)
  term3 <-  0.50*exp( (-(9*x1 - 7)^2/(4*vv[1])  - (9*x2 - 3)^2/(4*vv[2])  )*zz^2 )*sqrt(zz)
  term4 <- -0.20*exp( (-(9*x1 - 4)^2/(1*vv[1])  - (9*x2 - 7)^2/(1*vv[2])  )*zz^2 )*sqrt(zz)
  val <- term1 + term2 + term3 + term4
  return(val)

}



x2z3w1v2_ver00 <- function(x, z, w, v) {
  
  zAdd <- sum(z)
  if (w == 0) {
    val <- Matyas2D(x) + Sphere(v[1]) - .1
  }
  else if (w == 1) {
    val <- Matyas2D(x) + Sphere(v) - .3
  }
  else if (w == 2) {
    val <- Matyas2D(x) + Sphere(v[1]) + .1
  }
  else if (w == 3) {
    val <- Matyas2D(x) + Sphere(v)
  }  
  else {
    stop(sprintf('w = %d not defined', w))
  }
  return(val + zAdd)
  
}


x2z3w1v2_ver01 <- function(x, z, w, v) {
  
  zAdd <- sum(z)
  if (w == 0) {
    val <- Matyas2D(x) + Sphere(v[1]) - .1
  }
  else if (w == 1) {
    val <- Matyas2D(x) + Himmelblau2D(v) - .3
  }
  else if (w == 2) {
    val <- Matyas2D(x) + Sphere(v[1]) + .1
  }
  else if (w == 3) {
    val <- Matyas2D(x) + Himmelblau2D(v)
  }  
  else {
    stop(sprintf('w = %d not defined', w))
  }
  return(val + zAdd)
  
}



#library(globpso)
#res <- globpso(objFunc = hartmann3D, lower = c(0,0,0), upper = c(1,1,1))



