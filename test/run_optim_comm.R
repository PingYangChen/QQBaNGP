

runTest.BNGP <- function(TESTFUN, init_y, init_x, init_z, init_w, init_v, answerGrid, truthVal = NULL,
                         nSeq, contiParRange, categParRange, nSwarm, maxIter, nugget = 0., verbose = TRUE) {
  
  y <- init_y; x <- init_x; z <- init_z; w <- init_w; v <- init_v; n_init <- nrow(init_y)
  bnSeqData <- matrix( c(0, 0, min(y), NA, NA, NA, 0, 0), 1, 8)
  colnames(bnSeqData) <- c("n_acquisition", "rmse_fit", "y_min", "ei", "improvement", "uncertainty", 
                           "cpu_fit", "cpu_ei")
  for (i in 1:nSeq) {
    #i=1  
    bngp <- bngpFit(y, x, w, v, 
                    contiParRange, categParRange, 
                    nSwarm = nSwarm, maxIter = maxIter, nugget = nugget, optVerbose = verbose)
    #
    gridPrediction <- bngpPredict(bngp, answerGrid$x, answerGrid$w, answerGrid$v)
    rmse <- sqrt(mean((gridPrediction$pred - answerGrid$y)^2))
    #
    newpt <- bngpMaxEi(bngp, ei_alpha = 0.5, 
                       nSwarm = nSwarm, maxIter = maxIter, optVerbose = verbose)
    
    newpt$eiVal
    newpt$newpoint
    
    newptInfo <- bngpPredict(bngp, newpt$newpoint$x, newpt$newpoint$w, newpt$newpoint$v)
    
    x <- rbind(x, newpt$newpoint$x)
    z <- rbind(z, 0)
    w <- rbind(w, newpt$newpoint$w)
    for (vi in 1:length(v)) {
      v[[vi]] <- rbind(v[[vi]], newpt$newpoint$v[[vi]])
    }
    y <- rbind(y, TESTFUN(newpt$newpoint$x, newpt$newpoint$w, newpt$newpoint$v[[1]]) )
    
    bnSeqData <- rbind(bnSeqData, 
                       c(i, rmse, min(y), newptInfo$ei, newptInfo$improvement, newptInfo$uncertainty,
                         bngp$cputime, newpt$cputime))
    #
  }
  colnames(x) <- sprintf("x%02d", 1:ncol(x))
  colnames(w) <- sprintf("w%02d", 1:ncol(w))
  for (i in 1:length(v)) {
    colnames(v[[i]]) <- sprintf("%s^w%02d", sprintf("v%02d", 1:ncol(v[[i]])), i)
  }
  ct <- matrix(c(rep(0, n_init), 1:nSeq), n_init + nSeq, 1)
  colnames(ct) <- c("id")
  path <- cbind(ct, x, w)
  for (i in 1:length(v)) { path <- cbind(path, v[[i]]) }
  return(list(data = path, perf = bnSeqData))
}

runTest.GP <- function(TESTFUN, init_y, init_x, init_z, init_w, init_v, answerGrid, truthVal = NULL,
                       nSeq, contiParRange, categParRange, varParRange, nSwarm, maxIter, nugget = 0., verbose = TRUE) {
  
  y <- init_y; x <- init_x; z <- init_z; w <- init_w; v <- init_v; n_init <- nrow(init_y) 
  seqData <- matrix( c(0, 0, 0, NA, NA, NA, NA, min(y), NA, NA, 0, 0, NA, NA, NA, 0, 0), 1, 17)
  colnames(seqData) <- c("n_acquisition", "rmse_fit", "mape_fit", 
                         "true_loc_pred", "true_loc_ei", "true_loc_improvement", "true_loc_uncertainty",
                         "y_min", "y_add", "pred", "nbrmse", "nbmape", 
                         "ei", "improvement", "uncertainty", "cpu_fit", "cpu_ei")

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
  
  mdlList <- lapply(1:nSeq, function(k) {
    lapply(1:nrow(uz), function(zi) lapply(1:nrow(uw), function(wi) NULL))
  })
  
  for (i in 1:nSeq) {
    
    minVal <- min(y)
    out <- lapply(1:nrow(uz), function(zi) lapply(1:nrow(uw), function(wi) NULL))
    gridPrediction <- matrix(0, nrow(answerGrid$y), 1)
    for (zid in 1:nrow(uz)) {
      for (wid in 1:nrow(uw)) {
        
        zd <- z - matrix(uz[zid,], nrow(x), ncol(uz), byrow = TRUE)
        wd <- w - matrix(uw[wid,], nrow(x), ncol(uw), byrow = TRUE)
        loc <- which((zd == 0) & (wd == 0))
        
        suby <- y[loc,]
        subx <- x[loc,]
        subz <- z[loc,]
        subw <- w[loc,]
        subv <- lapply(1:wDim, function(k) v[[k]][loc,] )
        
        subXV <- subx
        for (k in 1:wDim) {
          vcol <- which(!is.na(subv[[k]][1,]))
          if (length(vcol) > 0) {
            subXV <- cbind(subXV, subv[[k]][,vcol])
          }
        }
        gp <- gpFit(suby, subXV, contiParRange, 
                    nSwarm = nSwarm, maxIter = maxIter, nugget = nugget, optVerbose = verbose)
        mdlList[[i]][[zid]][[wid]] <- gp
        #
        azd <- answerGrid$z - matrix(uz[zid,], nrow(answerGrid$z), ncol(uz), byrow = TRUE)
        awd <- answerGrid$w - matrix(uw[wid,], nrow(answerGrid$w), ncol(uw), byrow = TRUE)
        aloc <- which((azd == 0) & (awd == 0))
        ay <- answerGrid$y[aloc,]
        ax <- answerGrid$x[aloc,]
        az <- answerGrid$z[aloc,]
        aw <- answerGrid$w[aloc,]
        av <- lapply(1:wDim, function(k) answerGrid$v[[k]][aloc,] )
        aXV <- ax
        for (k in 1:wDim) {
          vcol <- which(!is.na(av[[k]][1,]))
          if (length(vcol) > 0) {
            aXV <- cbind(aXV, av[[k]][,vcol])
          }
        }
        gridPrediction[aloc,] <- gpPredict(gp, aXV)$pred
        #
        if (all((uz[zid,] - truthVal$z) == 0) & all((uw[wid,] - truthVal$w) == 0)) {
          print(c(zid, wid))
          tXV <- truthVal$x
          for (k in 1:wDim) {
            vcol <- which(!is.na(truthVal$v[[k]][1,]))
            if (length(vcol) > 0) {
              tXV <- cbind(tXV, matrix(truthVal$v[[k]][,vcol], 1, length(vcol)))
            }
          }
          truthPred <- gpPredict(gp, tXV)
        }
        #
        newpt <- gpMaxEi(gp, ei_alpha = 0.5, min_y = minVal,
                         nSwarm = nSwarm, maxIter = maxIter, optVerbose = verbose)
        newptInfo <- gpPredict(gp, newpt$newpoint$x)
        out[[zid]][[wid]] <- list(mdl = gp, ei = newpt, newptInfo = newptInfo)
      }
    }
    rmse <- sqrt(mean((gridPrediction - answerGrid$y)^2))
    mape <- 100*mean(abs(gridPrediction - answerGrid$y)/abs(answerGrid$y))
    #
    eiVal <- -Inf
    zwid <- NULL
    for (zid in 1:nrow(uz)) {
      for (wid in 1:nrow(uw)) {
        if (out[[zid]][[wid]]$ei$eiVal > eiVal) {
          eiVal <- out[[zid]][[wid]]$ei$eiVal
          zwid <- c(zid, wid)
        }
      }
    }
    gpTime <- sum(sapply(1:nrow(uz), function(zi) {
      sapply(1:nrow(uw), function(wi) { out[[zi]][[wi]]$mdl$cputime })
    }))
    eiTime <- sum(sapply(1:nrow(uz), function(zi) {
      sapply(1:nrow(uw), function(wi) { out[[zi]][[wi]]$ei$cputime })
    }))
    #
    nextpt <- out[[zwid[1]]][[zwid[2]]]$ei
    nextptInfo <- out[[zwid[1]]][[zwid[2]]]$newptInfo
    
    new_x <- matrix(nextpt$newpoint$x[,1:xDim], 1, xDim)
    new_z <- matrix(uz[zwid[1],], 1, zDim)
    new_w <- matrix(uw[zwid[2],], 1, wDim)
    new_vtmp <- matrix(nextpt$newpoint$x[,(xDim+1):ncol(nextpt$newpoint$x)], 1, ncol(nextpt$newpoint$x)-xDim)
    new_v <- lapply(1:wDim, function(k) matrix(NA, 1, vDim[k]))
    vct <- 1
    for (k in 1:wDim) {
      wloc <- which(uw[,k] == new_w[1,k])
      vloc <- which(vNaCol[[k]][wloc,] < 1)
      new_v[[k]][,vloc] <- new_vtmp[,vct:(vct+length(vloc) - 1)]
      vct <- vct + length(vloc)
    }
    new_y <- TESTFUN(new_x, new_z, new_w, new_v[[1]])
    #
    nbGrid <- 50; nbRange <- .03
    nbGrid_x <- matrix(runif(nbGrid*ncol(new_x), -nbRange, nbRange), nbGrid, ncol(new_x))
    gx <- matrix(new_x, nbGrid, ncol(new_x), byrow = TRUE) + nbGrid_x
    nbGrid_v <- matrix(runif(nbGrid*ncol(new_v[[1]]), -nbRange, nbRange), nbGrid, ncol(new_v[[1]]))
    gv <- list(matrix(new_v[[1]], nbGrid, ncol(new_v[[1]]), byrow = TRUE) + nbGrid_v)
    gx[gx < 0] <- 0; gx[gx > 1] <- 1
    gv[[1]][gv[[1]] < 0] <- 0; gv[[1]][gv[[1]] > 1] <- 1
    gXV <- gx
    for (k in 1:wDim) {
      vcol <- which(!is.na(gv[[k]][1,]))
      if (length(vcol) > 0) {
        gXV <- cbind(gXV, gv[[k]][,vcol])
      }
    }
    nbPrediction <- gpPredict(out[[zwid[1]]][[zwid[2]]]$mdl, gXV)
    gy <- matrix(0, nbGrid, 1)
    for (k in 1:nbGrid) {
      gy[k,] <- TESTFUN(gx[k,], new_z, new_w, gv[[1]][k,])
    }
    nbrmse <- sqrt(mean((nbPrediction$pred - gy)^2))
    nbmape <- 100*mean(abs(nbPrediction$pred - gy)/abs(gy))
    #
    x <- rbind(x, new_x)
    z <- rbind(z, new_z)
    w <- rbind(w, new_w)
    for (vi in 1:length(v)) {
      v[[vi]] <- rbind(v[[vi]], new_v[[vi]])
    }
    y <- rbind(y, new_y)
    #
    seqData <- rbind(seqData, 
                     c(i, rmse, mape, 
                       truthPred$pred, truthPred$ei, truthPred$improvement, truthPred$uncertainty,
                       min(y), new_y, nextptInfo$pred, nbrmse, nbmape, 
                       nextptInfo$ei, nextptInfo$improvement, nextptInfo$uncertainty,
                       gpTime, eiTime))
  }
  colnames(y) <- "y"
  colnames(x) <- sprintf("x%02d", 1:ncol(x))
  colnames(z) <- sprintf("z%02d", 1:ncol(z))
  colnames(w) <- sprintf("w%02d", 1:ncol(w))
  for (i in 1:length(v)) {
    colnames(v[[i]]) <- sprintf("%s^w%02d", sprintf("v%02d", 1:ncol(v[[i]])), i)
  }
  ct <- matrix(c(rep(0, n_init), 1:nSeq), n_init + nSeq, 1)
  colnames(ct) <- c("id")
  path <- cbind(ct, x, z, w)
  for (i in 1:length(v)) { path <- cbind(path, v[[i]]) }  
  path <- cbind(path, y)
  return(list(data = path, perf = seqData, mdl = mdlList))
}

runTest.QQBNGP <- function(TESTFUN, init_y, init_x, init_z, init_w, init_v, answerGrid, truthVal = NULL,
                           nSeq, contiParRange, categParRange, nSwarm, maxIter, nugget = 0., verbose = TRUE) {
  #verbose = TRUE
  y <- init_y; x <- init_x; z <- init_z; w <- init_w; v <- init_v; n_init <- nrow(init_y) 
  qqbnSeqData <- matrix( c(0, 0, 0, min(y), NA, NA, 0, 0, NA, NA, NA, 0, 0), 1, 13)
  colnames(qqbnSeqData) <- c("n_acquisition", "rmse_fit", "mape_fit", "y_min", "y_add", "pred", "nbrmse", "nbmape", 
                              "ei", "improvement", "uncertainty", "cpu_fit", "cpu_ei")
  for (i in 1:nSeq) {
    #i=1  
    qqbn <- qqbngpFit(y, x, z, w, v, 
                      contiParRange, categParRange, 
                      nSwarm = nSwarm, maxIter = maxIter, nugget = nugget, optVerbose = verbose)
    
    #round(abs(qqbngpPredict(qqbn, x, z, w, v)$pred - y), 6)
    gridPrediction <- qqbngpPredict(qqbn, answerGrid$x, answerGrid$z, answerGrid$w, answerGrid$v)
    rmse <- sqrt(mean((gridPrediction$pred - answerGrid$y)^2))
    mape <- 100*mean(abs(gridPrediction$pred - answerGrid$y)/abs(answerGrid$y))
    #
    # plot(gridPrediction$pred, answerGrid$y, 
    #      xlim = range(c(gridPrediction$pred, answerGrid$y)), ylim = range(c(gridPrediction$pred, answerGrid$y)))
    
    newpt <- qqbngpMaxEi(qqbn, ei_alpha = 0.5, 
                         nSwarm = nSwarm, maxIter = maxIter, optVerbose = verbose)
    # newpt$eiVal
    # newpt$newpoint
    
    newptInfo <- qqbngpPredict(qqbn, newpt$newpoint$x, newpt$newpoint$z, newpt$newpoint$w, newpt$newpoint$v)
    new_y <- TESTFUN(newpt$newpoint$x, newpt$newpoint$z, newpt$newpoint$w, newpt$newpoint$v[[1]])
    
    nbGrid <- 50; nbRange <- .03
    nbGrid_x <- matrix(runif(nbGrid*ncol(newpt$newpoint$x), -nbRange, nbRange), nbGrid, ncol(newpt$newpoint$x))
    gx <- matrix(newpt$newpoint$x, nbGrid, ncol(newpt$newpoint$x), byrow = TRUE) + nbGrid_x
    nbGrid_v <- matrix(runif(nbGrid*ncol(newpt$newpoint$v[[1]]), -nbRange, nbRange), nbGrid, ncol(newpt$newpoint$v[[1]]))
    gv <- list(matrix(newpt$newpoint$v[[1]], nbGrid, ncol(newpt$newpoint$v[[1]]), byrow = TRUE) + nbGrid_v)
    gx[gx < 0] <- 0; gx[gx > 1] <- 1
    gv[[1]][gv[[1]] < 0] <- 0; gv[[1]][gv[[1]] > 1] <- 1
    nbPrediction <- qqbngpPredict(qqbn, gx, matrix(newpt$newpoint$z, nbGrid, ncol(newpt$newpoint$z)), 
                                  matrix(newpt$newpoint$w, nbGrid, ncol(newpt$newpoint$w)), gv)
    gy <- matrix(0, nbGrid, 1)
    for (k in 1:nbGrid) {
      gy[k,] <- TESTFUN(gx[k,], newpt$newpoint$z, newpt$newpoint$w, gv[[1]][k,])
    }
    #plot(nbPrediction$pred, gy)
    nbrmse <- sqrt(mean((nbPrediction$pred - gy)^2))
    nbmape <- 100*mean(abs(nbPrediction$pred - gy)/abs(gy))
    
    x <- rbind(x, newpt$newpoint$x)
    z <- rbind(z, newpt$newpoint$z)
    w <- rbind(w, newpt$newpoint$w)
    for (vi in 1:length(v)) {
      v[[vi]] <- rbind(v[[vi]], newpt$newpoint$v[[vi]])
    }
    y <- rbind(y, new_y)
    
    qqbnSeqData <- rbind(qqbnSeqData, 
                         c(i, rmse, mape, min(y), new_y, newptInfo$pred, nbrmse, nbmape, 
                           newptInfo$ei, newptInfo$improvement, newptInfo$uncertainty,
                           qqbn$cputime, newpt$cputime))
    #
  }
  colnames(y) <- "y"
  colnames(x) <- sprintf("x%02d", 1:ncol(x))
  colnames(z) <- sprintf("z%02d", 1:ncol(z))
  colnames(w) <- sprintf("w%02d", 1:ncol(w))
  for (i in 1:length(v)) {
    colnames(v[[i]]) <- sprintf("%s^w%02d", sprintf("v%02d", 1:ncol(v[[i]])), i)
  }
  ct <- matrix(c(rep(0, n_init), 1:nSeq), n_init + nSeq, 1)
  colnames(ct) <- c("id")
  path <- cbind(ct, x, z, w)
  for (i in 1:length(v)) { path <- cbind(path, v[[i]]) }
  path <- cbind(path, y)
  return(list(data = path, perf = qqbnSeqData))
}

runTest.AQQBNGP <- function(TESTFUN, init_y, init_x, init_z, init_w, init_v, answerGrid, truthVal = NULL,
                            nSeq, contiParRange, categParRange, varParRange, nSwarm, maxIter, nugget = 0., 
                            verbose = TRUE) {
  
  y <- init_y; x <- init_x; z <- init_z; w <- init_w; v <- init_v; n_init <- nrow(init_y) 
  aqqbnSeqData <- matrix( c(0, 0, 0, NA, NA, NA, NA, min(y), NA, NA, 0, 0, NA, NA, NA, 0, 0), 1, 17)
  colnames(aqqbnSeqData) <- c("n_acquisition", "rmse_fit", "mape_fit", 
                              "true_loc_pred", "true_loc_ei", "true_loc_improvement", "true_loc_uncertainty",
                              "y_min", "y_add", "pred", "nbrmse", "nbmape", 
                              "ei", "improvement", "uncertainty", "cpu_fit", "cpu_ei")
  
  mdlList <- lapply(1:nSeq, function(k) NULL)
  for (i in 1:nSeq) {
    #i=1  
    aqqbn <- aqqbngpFit(y, x, z, w, v, 
                        contiParRange, categParRange, varParRange,
                        nSwarm = nSwarm, maxIter = maxIter, nugget = nugget, optVerbose = verbose)
    mdlList[[i]] <- aqqbn
    #round(abs(aqqbngpPredict(aqqbn, x, z, w, v)$pred - y), 6)
    gridPrediction <- aqqbngpPredict(aqqbn, answerGrid$x, answerGrid$z, answerGrid$w, answerGrid$v)
    rmse <- sqrt(mean((gridPrediction$pred - answerGrid$y)^2))
    mape <- 100*mean(abs(gridPrediction$pred - answerGrid$y)/abs(answerGrid$y))
    #
    # plot(gridPrediction$pred, answerGrid$y, 
    #      xlim = range(c(gridPrediction$pred, answerGrid$y)), ylim = range(c(gridPrediction$pred, answerGrid$y)))
    #
    truthPred <- aqqbngpPredict(aqqbn, truthVal$x, truthVal$z, truthVal$w, truthVal$v)
    #
    newpt <- aqqbngpMaxEi(aqqbn, ei_alpha = 0.5, 
                          nSwarm = nSwarm, maxIter = maxIter, optVerbose = verbose)
    # newpt$eiVal
    # newpt$newpoint
    
    newptInfo <- aqqbngpPredict(aqqbn, newpt$newpoint$x, newpt$newpoint$z, newpt$newpoint$w, newpt$newpoint$v)
    new_y <- TESTFUN(newpt$newpoint$x, newpt$newpoint$z, newpt$newpoint$w, newpt$newpoint$v[[1]])
    
    nbGrid <- 50; nbRange <- .03
    nbGrid_x <- matrix(runif(nbGrid*ncol(newpt$newpoint$x), -nbRange, nbRange), nbGrid, ncol(newpt$newpoint$x))
    gx <- matrix(newpt$newpoint$x, nbGrid, ncol(newpt$newpoint$x), byrow = TRUE) + nbGrid_x
    nbGrid_v <- matrix(runif(nbGrid*ncol(newpt$newpoint$v[[1]]), -nbRange, nbRange), nbGrid, ncol(newpt$newpoint$v[[1]]))
    gv <- list(matrix(newpt$newpoint$v[[1]], nbGrid, ncol(newpt$newpoint$v[[1]]), byrow = TRUE) + nbGrid_v)
    gx[gx < 0] <- 0; gx[gx > 1] <- 1
    gv[[1]][gv[[1]] < 0] <- 0; gv[[1]][gv[[1]] > 1] <- 1
    nbPrediction <- aqqbngpPredict(aqqbn, gx, matrix(newpt$newpoint$z, nbGrid, ncol(newpt$newpoint$z)), 
                                   matrix(newpt$newpoint$w, nbGrid, ncol(newpt$newpoint$w)), gv)
    gy <- matrix(0, nbGrid, 1)
    for (k in 1:nbGrid) {
      gy[k,] <- TESTFUN(gx[k,], newpt$newpoint$z, newpt$newpoint$w, gv[[1]][k,])
    }
    #plot(nbPrediction$pred, gy)
    nbrmse <- sqrt(mean((nbPrediction$pred - gy)^2))
    nbmape <- 100*mean(abs(nbPrediction$pred - gy)/abs(gy))
    
    x <- rbind(x, newpt$newpoint$x)
    z <- rbind(z, newpt$newpoint$z)
    w <- rbind(w, newpt$newpoint$w)
    for (vi in 1:length(v)) {
      v[[vi]] <- rbind(v[[vi]], newpt$newpoint$v[[vi]])
    }
    y <- rbind(y, new_y)
    
    aqqbnSeqData <- rbind(aqqbnSeqData, 
                          c(i, rmse, mape, 
                            truthPred$pred, truthPred$ei, truthPred$improvement, truthPred$uncertainty,
                            min(y), new_y, 
                            newptInfo$pred, nbrmse, nbmape, 
                            newptInfo$ei, newptInfo$improvement, newptInfo$uncertainty,
                            aqqbn$cputime, newpt$cputime))
    #
  }
  colnames(y) <- "y"
  colnames(x) <- sprintf("x%02d", 1:ncol(x))
  colnames(z) <- sprintf("z%02d", 1:ncol(z))
  colnames(w) <- sprintf("w%02d", 1:ncol(w))
  for (i in 1:length(v)) {
    colnames(v[[i]]) <- sprintf("%s^w%02d", sprintf("v%02d", 1:ncol(v[[i]])), i)
  }
  ct <- matrix(c(rep(0, n_init), 1:nSeq), n_init + nSeq, 1)
  colnames(ct) <- c("id")
  path <- cbind(ct, x, z, w)
  for (i in 1:length(v)) { path <- cbind(path, v[[i]]) }
  path <- cbind(path, y)
  return(list(data = path, perf = aqqbnSeqData, mdl = mdlList))
}
