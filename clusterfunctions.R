lbart.clusterseed <- function(seed = NULL, x.train, y.train, x.test=matrix(0.0,0,0),
                              sparse=FALSE, a=0.5, b=1, augment=FALSE, rho=NULL,
                              xinfo=matrix(0.0,0,0), usequants=FALSE,
                              cont=FALSE, rm.const=TRUE, tau.interval=0.95,
                              k=2.0, power=2.0, base=.95,
                              binaryOffset=NULL,
                              ntree=200L, numcut=100L,
                              ndpost=1000L, nskip=100L,
                              keepevery=1L,
                              nkeeptrain=ndpost, nkeeptest=ndpost,
                              #nkeeptestmean=ndpost,
                              nkeeptreedraws=ndpost,
                              printevery=100, transposed=FALSE) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  BART::lbart(x.train=x.train, y.train=y.train, x.test=x.test,
        sparse=sparse, a=a, b=b, augment=augment, rho=rho,
        xinfo=xinfo, tau.interval=tau.interval,
        k=k, power=power, base=base,
        binaryOffset=binaryOffset,
        ntree=ntree, numcut=numcut,
        ndpost=ndpost, nskip=nskip, keepevery=keepevery,
        ## nkeeptrain=mc.nkeep, nkeeptest=mc.nkeep,
        ## nkeeptestmean=mc.nkeep, nkeeptreedraws=mc.nkeep,
        printevery=printevery, transposed=transposed)
}


lbart.cluster <- function(x.train, y.train, x.test=matrix(0.0,0,0),
                          sparse=FALSE, a=0.5, b=1, augment=FALSE, rho=NULL,
                          xinfo=matrix(0.0,0,0), usequants=FALSE,
                          cont=FALSE, rm.const=TRUE, tau.interval=0.95,
                          k=2.0, power=2.0, base=.95,
                          binaryOffset=NULL,
                          ntree=200L, numcut=100L,
                          ndpost=1000L, nskip=100L,
                          keepevery=1L,
                          nkeeptrain=ndpost, nkeeptest=ndpost,
                          #nkeeptestmean=ndpost,
                          nkeeptreedraws=ndpost,
                          printevery=100, transposed=FALSE, nchains = 2) {
  
  yhat.train.mat <- array(NA, dim = c(ndpost / keepevery,
                                      dim(x.train)[transposed + 1], nchains))
  yhat.test.mat <- array(NA, dim = c(ndpost / keepevery,
                                     dim(x.test)[transposed + 1], nchains))
  
  cl <- parallel::makeCluster(nchains)
  on.exit(parallel::stopCluster(cl), add = TRUE, after = FALSE)
  seeds <- sample(1:10000, size = nchains)
  res <- do.call(parallel::parLapply, list(X = seeds, cl = cl, fun = lbart.clusterseed,
                                        
                                        x.train=x.train, y.train=y.train, x.test=x.test,
                                        sparse=sparse, a=a, b=b, augment=augment, rho=rho,
                                        xinfo=xinfo, tau.interval=tau.interval,
                                        k=k, power=power, base=base,
                                        binaryOffset=binaryOffset,
                                        ntree=ntree, numcut=numcut,
                                        ndpost=ndpost, nskip=nskip, keepevery=keepevery,
                                        ## nkeeptrain=mc.nkeep, nkeeptest=mc.nkeep,
                                        ## nkeeptestmean=mc.nkeep, nkeeptreedraws=mc.nkeep,
                                        printevery=printevery, transposed=transposed))
  # Unpack
  for (i in 1:nchains) {
    currentchain <- res[[i]]
    yhat.train.mat[, , i] <- currentchain$yhat.train
    yhat.test.mat[, , i] <- currentchain$yhat.test
  }
  
  return(list("yhat.train" = yhat.train.mat, "yhat.test" = yhat.test.mat))
  
}


wbart.clusterseed <- function(seed = NULL, x.train, y.train, x.test=matrix(0.0,0,0),
                              sparse=FALSE, theta=0, omega=1,
                              a=0.5, b=1, augment=FALSE, rho=NULL,
                              xinfo=matrix(0.0,0,0), usequants=FALSE,
                              cont=FALSE, rm.const=TRUE,
                              sigest=NA, sigdf=3, sigquant=.90,
                              k=2.0, power=2.0, base=.95,
                              sigmaf=NA, lambda=NA,
                              fmean=mean(y.train),
                              w=rep(1,length(y.train)),
                              ntree=200L, numcut=100L,
                              ndpost=1000L, nskip=100L, keepevery=1L,
                              nkeeptrain=ndpost, nkeeptest=ndpost,
                              nkeeptestmean=ndpost, nkeeptreedraws=ndpost,
                              printevery=100L, transposed=FALSE) {
  if (!is.null(seed)) {
    set.seed(seed)
  }
  BART::wbart(x.train=x.train, y.train=y.train, x.test=x.test,
        sparse=sparse, theta=theta, omega=omega,
        a=a, b=b, augment=augment, rho=rho,
        xinfo=xinfo,
        sigest=sigest, sigdf=sigdf, sigquant=sigquant,
        k=k, power=power, base=base,
        sigmaf=sigmaf, lambda=lambda, fmean=fmean, w=w,
        ntree=ntree, numcut=numcut,
        ndpost=ndpost, nskip=nskip, keepevery=keepevery,
        printevery=printevery, transposed=transposed)
}

wbart.cluster <- function(x.train, y.train, x.test=matrix(0.0,0,0),
                          sparse=FALSE, theta=0, omega=1,
                          a=0.5, b=1, augment=FALSE, rho=NULL,
                          xinfo=matrix(0.0,0,0), usequants=FALSE,
                          cont=FALSE, rm.const=TRUE,
                          sigest=NA, sigdf=3, sigquant=.90,
                          k=2.0, power=2.0, base=.95,
                          sigmaf=NA, lambda=NA,
                          fmean=mean(y.train),
                          w=rep(1,length(y.train)),
                          ntree=200L, numcut=100L,
                          ndpost=1000L, nskip=100L, keepevery=1L,
                          nkeeptrain=ndpost, nkeeptest=ndpost,
                          nkeeptestmean=ndpost, nkeeptreedraws=ndpost,
                          printevery=100L, transposed=FALSE, nchains = 2) {
  
  sigma.mat <- array(NA, dim = c(ndpost + nskip, nchains))
  yhat.train.mat <- array(NA, dim = c(ndpost / keepevery,
                                      dim(x.train)[transposed + 1], nchains))
  yhat.test.mat <- array(NA, dim = c(ndpost / keepevery,
                                     dim(x.test)[transposed + 1], nchains))
  
  cl <- parallel::makeCluster(nchains)
  on.exit(parallel::stopCluster(cl), add = TRUE, after = FALSE)
  seeds <- sample(1:10000, size = nchains)
  res <- do.call(parallel::parLapply, list(X = seeds, cl = cl, fun = wbart.clusterseed,
                                           
                                           x.train=x.train, y.train=y.train, x.test=x.test,
                                           sparse=sparse, theta=theta, omega=omega,
                                           a=a, b=b, augment=augment, rho=rho,
                                           xinfo=xinfo,
                                           sigest=sigest, sigdf=sigdf, sigquant=sigquant,
                                           k=k, power=power, base=base,
                                           sigmaf=sigmaf, lambda=lambda, fmean=fmean, w=w,
                                           ntree=ntree, numcut=numcut,
                                           ndpost=ndpost, nskip=nskip, keepevery=keepevery,
                                           printevery=printevery, transposed=transposed))
  # Unpack
  for (i in 1:nchains) {
    currentchain <- res[[i]]
    sigma.mat[, i] <- currentchain$sigma
    yhat.train.mat[, , i] <- currentchain$yhat.train
    yhat.test.mat[, , i] <- currentchain$yhat.test
  }

  
  return(list("sigma" = sigma.mat, "yhat.train" = yhat.train.mat,
              "yhat.test" = yhat.test.mat))
  
}
