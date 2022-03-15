
library(Rcpp)
library(RcppArmadillo)
library(globpso)
gdpath <- 'D:\\PYChen\\GoogleDrive\\PYChen_Statistics_NCKU\\Researches\\2021_QQBaN'
codePath <- file.path(gdpath, 'code\\qqbngp')

cPath <- file.path(codePath, 'src')
sourceCpp(file.path(cPath, 'cppFunc.cpp'))


################################################################################
### GP
################################################################################
# y: vector
# x: matrix
# w: integer matrix
# v: list of matrices of length equals to ncol(w)
gpFit <- function(y, x, 
                  contiParRange = 10^c(-3, .5), 
                  nSwarm = 64, maxIter = 200, nugget = 0., optVerbose = TRUE) {
  
  cputime <- system.time({
    xDim <- ncol(x)
    nContiPar <- xDim
    
    low_bound <- rep(min(contiParRange), nContiPar)
    upp_bound <- rep(max(contiParRange), nContiPar)
    
    alg_setting <- getPSOInfo(nSwarm = nSwarm, maxIter = maxIter, psoType = "quantum")
    
    res <- globpso(objFunc = gpObjCpp, lower = low_bound, upper = upp_bound,
                   PSO_INFO = alg_setting, verbose = optVerbose,
                   y = y, x = x, nugget = nugget)
    #res$val
    mdl <- gpModel(param = res$par, y = y, x = x, nugget = nugget)
    mdl$data <- list(y = y, x = x, xDim = xDim)
  })[3]
  mdl$cputime <- cputime
  cat(sprintf("GP FIT CPU time: %.2f seconds.\n", cputime))
  return(mdl)
}

gpPredict <- function(gp, x0, ei_alpha = 0.5, min_y = NULL) {
  
  if (is.null(min_y)) { min_y <- min(gp$data$y) }
  pred <- gpPred(x0, gp$data$y, gp$data$x, 
                 gp$alpha, gp$invPsi, gp$mu, gp$sigma, ei_alpha, min_y)
  return(pred)
} 


gpMaxEi <- function(gp, ei_alpha = 0.5, min_y = NULL, nSwarm = 64, maxIter = 200, optVerbose = TRUE) {
  
  if (is.null(min_y)) { min_y <- min(gp$data$y) }
  cputime <- system.time({
    alg_setting <- getPSOInfo(nSwarm = nSwarm, maxIter = maxIter, psoType = "quantum")
    #
    low_bound <- rep(0, gp$data$xDim)
    upp_bound <- rep(1, gp$data$xDim)
    #
    res <- globpso(objFunc = gpMaxEiObj, lower = low_bound, upper = upp_bound,
                   PSO_INFO = alg_setting, verbose = optVerbose,
                   gp = gp, ei_alpha = ei_alpha, min_y = min_y)
    
    rx <- matrix(res$par, 1, gp$data$xDim)
    rdata <- list(x = rx)
  })[3]
  cat(sprintf("BNGP MAXEI CPU time: %.2f seconds.\n", cputime))
  return(list(eiVal = exp(-res$val),
              newpoint = rdata,
              cputime = cputime))
}

gpMaxEiObj <- function(xx, gp, ei_alpha = 0.5, min_y = NULL) {

  if (is.null(min_y)) { min_y <- min(gp$data$y) }
  pred <- gpPred(xx, gp$data$y, gp$data$x, 
                 gp$alpha, gp$invPsi, gp$mu, gp$sigma, ei_alpha, min_y)
  
  return( -log(pred$ei[1,1]) )
}




################################################################################
### BNGP
################################################################################
# y: vector
# x: matrix
# w: integer matrix
# v: list of matrices of length equals to ncol(w)
bngpFit <- function(y, x, w, v, 
                    contiParRange = 10^c(-3, .5), 
                    categParRange = c(0.15, 0.5), 
                    nSwarm = 64, maxIter = 200, nugget = 0., optVerbose = TRUE) {
  
  cputime <- system.time({
    xDim <- ncol(x)
    wDim <- ncol(w)
    vDim <- sapply(1:wDim, function(i) ncol(v[[i]]))
    wlvs <- lapply(1:wDim, function(i) unique(w[,i]))
    nwlv <- sapply(1:wDim, function(i) length(unique(w[,i])))
    nContiPar <- xDim + sum(vDim*nwlv)
    nCategPar <- wDim
    
    uw <- unique(w)

    vNaCol <- lapply(1:wDim, function(i) matrix(0, nwlv[i], vDim[[i]]))
    for (j in 1:wDim) {
      uwj <- unique(w[,j])
      for (u in 1:length(uwj)) {
        rid <- which(w[,j] == uwj[u])[1]  
        vNaCol[[j]][u,] <- is.na(v[[j]][rid,])
      }
    }
    
    low_bound <- c(rep(min(contiParRange), nContiPar), 
                   rep(min(categParRange), nCategPar))
    upp_bound <- c(rep(max(contiParRange), nContiPar), 
                   rep(max(categParRange), nCategPar))
    
    alg_setting <- getPSOInfo(nSwarm = nSwarm, maxIter = maxIter, psoType = "quantum")
    
    res <- globpso(objFunc = bngpObjCpp, lower = low_bound, upper = upp_bound,
                   PSO_INFO = alg_setting, verbose = optVerbose,
                   y = y, x = x, w = w, v = v, nugget = nugget)
    #res$val
    mdl <- bngpModel(param = res$par, y = y, x = x, w = w, v = v, nugget = nugget)
    mdl$data <- list(y = y, x = x, w = w, v = v, 
                     xDim = xDim, wDim = wDim, nwlv = nwlv, wlvs = wlvs, uw = uw, 
                     vDim = vDim, vNaCol = vNaCol)
  })[3]
  mdl$cputime <- cputime
  cat(sprintf("BNGP FIT CPU time: %.2f seconds.\n", cputime))
  return(mdl)
}

bngpPredict <- function(bngp, x0, w0, v0, ei_alpha = 0.5) {
  
  pred <- bngpPred(x0, w0, v0, bngp$data$y, bngp$data$x, bngp$data$w, bngp$data$v, 
                   bngp$vecParams, bngp$invPsi, bngp$mu, bngp$sigma, ei_alpha)
  return(pred)
} 


bngpMaxEi <- function(bngp, ei_alpha = 0.5, nSwarm = 64, maxIter = 200, optVerbose = TRUE) {
  
  cputime <- system.time({
    alg_setting <- getPSOInfo(nSwarm = nSwarm, maxIter = maxIter, psoType = "quantum")
    #
    low_bound <- rep(0, bngp$data$xDim + sum(bngp$data$vDim))
    upp_bound <- rep(1, bngp$data$xDim + sum(bngp$data$vDim))
    #
    eiPars <- matrix(0, nrow(bngp$data$uw), length(low_bound))
    eiVals <- rep(0, nrow(bngp$data$uw))
    for (u in 1:nrow(bngp$data$uw)) {
      
      vNaColVec <- NULL
      for (j in 1:bngp$data$wDim) { 
        wlvid <- which(bngp$data$wlvs[[j]] == bngp$data$uw[u,j])
        vNaColVec <- c(vNaColVec, bngp$data$vNaCol[[j]][wlvid,])
      }
      ub <- upp_bound
      if (any(vNaColVec == 1)) {
        ub[bngp$data$xDim + which(vNaColVec == 1)] <- 0  
      }
      #
      res <- globpso(objFunc = bngpMaxEiObj, lower = low_bound, upper = ub,
                     PSO_INFO = alg_setting, verbose = optVerbose,
                     ww = bngp$data$uw[u,], bngp = bngp, ei_alpha = ei_alpha)
      
      eiPars[u,] <- res$par
      eiVals[u] <- res$val
    }
    bestContiVec <- eiPars[which.min(eiVals),]
    rx <- matrix(bestContiVec[1:bngp$data$xDim], 1, bngp$data$xDim)
    rw <- matrix(bngp$data$uw[which.min(eiVals),], 1, bngp$data$wDim) 
    rv <- lapply(1:length(bngp$data$vDim), function(i) {})
    #
    ct <- bngp$data$xDim
    for (i in 1:bngp$data$wDim) {
      rv[[i]] <- matrix(bestContiVec[(ct+1):(ct+bngp$data$vDim[i])], 1, bngp$data$vDim[i])
      wlvid <- which(bngp$data$wlvs[[i]] == rw[,i])
      if (any(bngp$data$vNaCol[[i]][wlvid,] == 1)) {
        rv[[i]][,which(bngp$data$vNaCol[[i]][wlvid,] == 1)] <- NA  
      }
      ct <- ct + bngp$data$vDim[i]
    }
    rdata <- list(x = rx, w = rw, v = rv)
  })[3]
  cat(sprintf("BNGP MAXEI CPU time: %.2f seconds.\n", cputime))
  return(list(eiVal = exp(-eiVals[which.min(eiVals)]),
              newpoint = rdata,
              cputime = cputime))
}

bngpMaxEiObj <- function(contiVec, ww, bngp, ei_alpha = 0.5) {

  ww <- matrix(ww, 1, bngp$data$wDim)
  xx <- matrix(contiVec[1:bngp$data$xDim], 1, bngp$data$xDim)
  vv <- lapply(1:length(bngp$data$vDim), function(i) {})
  ct <- bngp$data$xDim
  for (i in 1:length(bngp$data$vDim)) {
    vv[[i]] <- matrix(contiVec[(ct+1):(ct+bngp$data$vDim[i])], 1, bngp$data$vDim[i])
    ct <- ct + bngp$data$vDim[i]
  }
  #  
  pred <- bngpPred(xx, ww, vv, bngp$data$y, bngp$data$x, bngp$data$w, bngp$data$v, 
                   bngp$vecParams, bngp$invPsi, bngp$mu, bngp$sigma, ei_alpha)
  
  return( -log(pred$ei[1,1]) )
}


################################################################################
###
################################################################################
# y: vector
# x: matrix
# z: integer matrix
# w: integer matrix
# v: list of matrices of length equals to ncol(w)
qqbngpFit <- function(y, x, z, w, v, 
                      contiParRange = 10^c(-3, .5), 
                      categParRange = c(0.15, 0.5), 
                      nSwarm = 64, maxIter = 200, nugget = 0., optVerbose = TRUE) {
  xDim <- ncol(x)
  zDim <- ncol(z)
  wDim <- ncol(w)
  vDim <- sapply(1:wDim, function(i) ncol(v[[i]]))
  zlv <- sapply(1:ncol(z), function(i) length(unique(z[,i])))
  wlv <- sapply(1:ncol(w), function(i) length(unique(w[,i])))
  
  nContiPar <- xDim + sum(vDim*wlv)
  nCategPar <- zDim + wDim
  
  low_bound <- c(rep(min(contiParRange), nContiPar), 
                 rep(min(categParRange), nCategPar))
  upp_bound <- c(rep(max(contiParRange), nContiPar), 
                 rep(max(categParRange), nCategPar))
  
  alg_setting <- getPSOInfo(nSwarm = nSwarm, maxIter = maxIter, psoType = "quantum")
  
  res <- globpso(objFunc = qqbngpObjCpp, lower = low_bound, upper = upp_bound,
                 PSO_INFO = alg_setting, verbose = optVerbose,
                 y = y, x = x, z = z, w = w, v = v, nugget = nugget)
  #res$val
  mdl <- qqbngpModel(param = res$par, y = y, x = x, z = z, w = w, v = v, nugget = nugget)
  mdl$data <- list(y = y, x = x, z = z, w = w, v = v)
  return(mdl)
}

qqbngpPredict <- function(qqbngp, x0, z0, w0, v0, ei_alpha = 0.5, min_y = NULL) {
  
  if (is.null(min_y)) { min_y <- min(qqbngp$data$y) }
  pred <- qqbngpPred(x0, z0, w0, v0, 
                     qqbngp$data$y, qqbngp$data$x, qqbngp$data$z, qqbngp$data$w, qqbngp$data$v, 
                     qqbngp$vecParams, qqbngp$invPsi, qqbngp$mu, qqbngp$sigma, ei_alpha, min_y)
  return(pred)
} 

qqbngpMaxEi <- function(qqbngp, ei_alpha = 0.5, min_y = NULL, nSwarm = 64, maxIter = 200, optVerbose = TRUE) {
  
  if (is.null(min_y)) { min_y <- min(qqbngp$data$y) }
  alg_setting <- getPSOInfo(nSwarm = nSwarm, maxIter = maxIter, psoType = "quantum")
  
  xDim <- ncol(qqbngp$data$x)
  zDim <- ncol(qqbngp$data$z)
  wDim <- ncol(qqbngp$data$w)
  vDim <- sapply(1:wDim, function(i) ncol(qqbngp$data$v[[i]]))
  low_bound <- rep(0, xDim + sum(vDim))
  upp_bound <- rep(1, xDim + sum(vDim))
  #
  uzw <- unique(cbind(qqbngp$data$z, qqbngp$data$w))
  eiPars <- matrix(0, nrow(uzw), length(low_bound))
  eiVals <- rep(0, nrow(uzw))
  for (u in 1:nrow(uzw)) {
    
    res <- globpso(objFunc = qqbngpMaxEiObj, lower = low_bound, upper = upp_bound,
                   PSO_INFO = alg_setting, verbose = optVerbose,
                   xDim = xDim, zDim = zDim, wDim = wDim, vDim = vDim, zw = uzw[u,],
                   qqbngp = qqbngp, ei_alpha = ei_alpha, min_y = min_y)
    
    eiPars[u,] <- res$par
    eiVals[u] <- res$val
  }
  bestContiVec <- eiPars[which.min(eiVals),]
  rx <- bestContiVec[1:xDim]
  rv <- lapply(1:length(vDim), function(i) {})
  ct <- xDim
  for (i in 1:length(vDim)) {
    rv[[i]] <- matrix(bestContiVec[(ct+1):(ct+vDim[i])], 1, vDim[i])
    ct <- ct + vDim[i]
  }
  rdata <- list(
    x = matrix(rx, 1, xDim),
    z = matrix(uzw[which.min(eiVals),1:zDim], 1, zDim),
    w = matrix(uzw[which.min(eiVals),(zDim+1):(zDim+wDim)], 1, wDim),
    v = rv
  )
  return(list(eiVal = exp(-eiVals[which.min(eiVals)]),
              newpoint = rdata))
}

qqbngpMaxEiObj <- function(contiVec, xDim, zDim, wDim, vDim, zw, qqbngp, ei_alpha = 0.5, min_y = NULL) {
  
  zz <- matrix(zw[1:zDim], 1, wDim)
  ww <- matrix(zw[(zDim+1):(zDim+wDim)], 1, wDim)
  xx <- matrix(contiVec[1:xDim], 1, xDim)
  vv <- lapply(1:length(vDim), function(i) {})
  ct <- xDim
  for (i in 1:length(vDim)) {
    vv[[i]] <- matrix(contiVec[(ct+1):(ct+vDim[i])], 1, vDim[i])
    ct <- ct + vDim[i]
  }
  #
  if (is.null(min_y)) { min_y <- min(qqbngp$data$y) }
  pred <- qqbngpPred(xx, zz, ww, vv, qqbngp$data$y, qqbngp$data$x, qqbngp$data$z, qqbngp$data$w, qqbngp$data$v, 
                     qqbngp$vecParams, qqbngp$invPsi, qqbngp$mu, qqbngp$sigma, ei_alpha, min_y)
  
  return( -log(pred$ei[1,1]) )
}



################################################################################
###
################################################################################aqqbngpFit <- function(y, x, z, w, v, 
# y: vector
# x: matrix
# z: integer matrix
# w: integer matrix
# v: list of matrices of length equals to ncol(w)
aqqbngpFit <- function(y, x, z, w, v,                        
                       contiParRange = 10^c(-3, .5), 
                       categParRange = c(0.15, 0.5), 
                       varParRange = 10^c(-3, .5),
                       nSwarm = 64, maxIter = 200, nugget = 0., optVerbose = TRUE) {
  # x #(alpha) = xDim*(zDim + wDim)
  # v #(gammas) = SUM( vDim(u)*wlv(u) )
  # z #(taus) = SUM( choose(zlv(u), 2)) 
  # w #(thetas) = SUM( choose(wlv(u), 2)) 
  # vz #(sigmas) = zDim
  # vw #(nus) = wDim
  # vwu #(upsilons) = SUM( wlv(u) )
  cputime <- system.time({
    xDim <- ncol(x)
    zDim <- ncol(z)
    wDim <- ncol(w)
    vDim <- sapply(1:wDim, function(i) ncol(v[[i]]))
    zlvs <- lapply(1:zDim, function(i) unique(z[,i]))
    wlvs <- lapply(1:wDim, function(i) unique(w[,i]))
    nzlv <- sapply(1:ncol(z), function(i) length(unique(z[,i])))
    nwlv <- sapply(1:ncol(w), function(i) length(unique(w[,i])))
    nzlv2 <- sapply(1:length(nzlv), function(i) floor(nzlv[i]*(nzlv[i] - 1)/2))
    nwlv2 <- sapply(1:length(nwlv), function(i) floor(nwlv[i]*(nwlv[i] - 1)/2))
    
    uzw <- unique(cbind(z, w))
    uz <- unique(z)
    uw <- unique(w)
    
    vNaCol <- lapply(1:wDim, function(i) matrix(0, nwlv[i], vDim[[i]]))
    for (j in 1:wDim) {
      uwj <- unique(w[,j])
      for (u in 1:length(uwj)) {
        rid <- which(w[,j] == uwj[u])[1]  
        vNaCol[[j]][u,] <- is.na(v[[j]][rid,])
      }
    }
    
    nContiPar <- xDim*(zDim + wDim) + sum(vDim*nwlv)
    nCategPar <- sum(nzlv2) + sum(nwlv2) 
    nVarPar <- zDim + wDim + sum(nwlv)
    
    low_bound <- c(rep(min(contiParRange), nContiPar), 
                   rep(min(categParRange), nCategPar), 
                   rep(min(varParRange), nVarPar))
    upp_bound <- c(rep(max(contiParRange), nContiPar), 
                   rep(max(categParRange), nCategPar), 
                   rep(max(varParRange), nVarPar))
    
    alg_setting <- getPSOInfo(nSwarm = nSwarm, maxIter = maxIter, psoType = "quantum")
    
    res <- globpso(objFunc = aqqbngpObjCpp, lower = low_bound, upper = upp_bound,
                   PSO_INFO = alg_setting, verbose = optVerbose,
                   y = y, x = x, z = z, w = w, v = v, nugget = nugget)
    #res$val
    mdl <- aqqbngpModel(param = res$par, y = y, x = x, z = z, w = w, v = v, nugget = nugget)
    mdl$data <- list(y = y, x = x, z = z, w = w, v = v,
                     xDim = xDim, zDim = zDim, wDim = wDim, 
                     nzlv = nzlv, zlvs = zlvs, uz = uz, 
                     nwlv = nwlv, wlvs = wlvs, uw = uw, uzw = uzw,
                     vDim = vDim, vNaCol = vNaCol)
  })[3]
  mdl$cputime <- cputime
  cat(sprintf("AQQBNGP FIT CPU time: %.2f seconds.\n", cputime))
  return(mdl)
}

aqqbngpPredict <- function(aqqbngp, x0, z0, w0, v0, ei_alpha = 0.5, min_y = NULL) {
  
  if (is.null(min_y)) { min_y <- min(aqqbngp$data$y) }
  pred <- aqqbngpPred(x0, z0, w0, v0, aqqbngp$data$y, aqqbngp$data$x, aqqbngp$data$z, 
                      aqqbngp$data$w, aqqbngp$data$v, aqqbngp$vecParams, aqqbngp$invPsi, aqqbngp$mu,
                      ei_alpha, min_y)
  return(pred)
} 

aqqbngpMaxEi <- function(aqqbngp, ei_alpha = 0.5, min_y = NULL, nSwarm = 64, maxIter = 200, optVerbose = TRUE) {
  
  if (is.null(min_y)) { min_y <- min(aqqbngp$data$y) }
  cputime <- system.time({
    alg_setting <- getPSOInfo(nSwarm = nSwarm, maxIter = maxIter, psoType = "quantum")
    
    low_bound <- rep(0, aqqbngp$data$xDim + sum(aqqbngp$data$vDim))
    upp_bound <- rep(1, aqqbngp$data$xDim + sum(aqqbngp$data$vDim))
    #
    eiPars <- matrix(0, nrow(aqqbngp$data$uzw), length(low_bound))
    eiVals <- rep(0, nrow(aqqbngp$data$uzw))
    for (u in 1:nrow(aqqbngp$data$uzw)) {
      
      vNaColVec <- NULL
      for (j in 1:aqqbngp$data$wDim) { 
        wlvid <- which(aqqbngp$data$wlvs[[j]] == aqqbngp$data$uzw[u,aqqbngp$data$zDim+j])
        vNaColVec <- c(vNaColVec, aqqbngp$data$vNaCol[[j]][wlvid,])
      }
      ub <- upp_bound
      if (any(vNaColVec == 1)) {
        ub[aqqbngp$data$xDim + which(vNaColVec == 1)] <- 0  
      }
      
      res <- globpso(objFunc = aqqbngpMaxEiObj, lower = low_bound, upper = ub,
                     PSO_INFO = alg_setting, verbose = optVerbose,
                     zw = aqqbngp$data$uzw[u,], aqqbngp = aqqbngp, ei_alpha = ei_alpha, min_y = min_y)
      
      eiPars[u,] <- res$par
      eiVals[u] <- res$val
    }
    bestContiVec <- eiPars[which.min(eiVals),]
    rx <- matrix(bestContiVec[1:aqqbngp$data$xDim], 1, aqqbngp$data$xDim)
    rz <- matrix(aqqbngp$data$uzw[which.min(eiVals),1:aqqbngp$data$zDim], 1, aqqbngp$data$zDim)
    rw <- matrix(aqqbngp$data$uzw[which.min(eiVals),(aqqbngp$data$zDim+1):(aqqbngp$data$zDim+aqqbngp$data$wDim)], 1, aqqbngp$data$wDim)
    rv <- lapply(1:length(aqqbngp$data$vDim), function(i) {})
    ct <- aqqbngp$data$xDim
    for (i in 1:aqqbngp$data$wDim) {
      rv[[i]] <- matrix(bestContiVec[(ct+1):(ct+aqqbngp$data$vDim[i])], 1, aqqbngp$data$vDim[i])
      wlvid <- which(aqqbngp$data$wlvs[[i]] == rw[,i])
      if (any(aqqbngp$data$vNaCol[[i]][wlvid,] == 1)) {
        rv[[i]][,which(aqqbngp$data$vNaCol[[i]][wlvid,] == 1)] <- NA  
      }
      ct <- ct + aqqbngp$data$vDim[i]
    }
    rdata <- list( x = rx, z = rz, w = rw, v = rv )
  })[3]
  cat(sprintf("AQQBNGP MAXEI CPU time: %.2f seconds.\n", cputime))
  return(list(eiVal = exp(-eiVals[which.min(eiVals)]),
              newpoint = rdata,
              cputime = cputime))
}

aqqbngpMaxEiObj <- function(contiVec, zw, aqqbngp, ei_alpha = 0.5, min_y = NULL) {
  
  zz <- matrix(zw[1:aqqbngp$data$zDim], 1, aqqbngp$data$zDim)
  ww <- matrix(zw[(aqqbngp$data$zDim+1):(aqqbngp$data$zDim+aqqbngp$data$wDim)], 1, aqqbngp$data$wDim)
  xx <- matrix(contiVec[1:aqqbngp$data$xDim], 1, aqqbngp$data$xDim)
  vv <- lapply(1:length(aqqbngp$data$vDim), function(i) {})
  ct <- aqqbngp$data$xDim
  for (i in 1:length(aqqbngp$data$vDim)) {
    vv[[i]] <- matrix(contiVec[(ct+1):(ct+aqqbngp$data$vDim[i])], 1, aqqbngp$data$vDim[i])
    ct <- ct + aqqbngp$data$vDim[i]
  }
  #  
  if (is.null(min_y)) { min_y <- min(aqqbngp$data$y) }
  pred <- aqqbngpPred(xx, zz, ww, vv, aqqbngp$data$y, aqqbngp$data$x, aqqbngp$data$z, 
                      aqqbngp$data$w, aqqbngp$data$v, aqqbngp$vecParams, aqqbngp$invPsi, aqqbngp$mu,
                      ei_alpha, min_y)
  
  return( -log(pred$ei[1,1]) )
}



