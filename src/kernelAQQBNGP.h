
// HEADER
double aqqbnCorrKern(const arma::rowvec &xi, const arma::urowvec &zi, const arma::urowvec &wi, const arma::field<arma::rowvec> &vi, 
                     const arma::rowvec &xj, const arma::urowvec &zj, const arma::urowvec &wj, const arma::field<arma::rowvec> &vj, 
                     const arma::mat &alpha, const arma::field<arma::mat> &taus, 
                     const arma::field<arma::mat> &thetas, const arma::field<arma::mat> &gammas,
                     const arma::vec &sigmas, const arma::vec &nus, const arma::field<arma::mat> &upsilons);

void aqqbnCorrMat(arma::mat &psi, const arma::mat &x, const arma::umat &z, const arma::umat &w, const arma::field<arma::mat> &v, 
                  const arma::mat &alpha, const arma::field<arma::mat> &taus, 
                  const arma::field<arma::mat> &thetas, const arma::field<arma::mat> &gammas,
                  const arma::vec &sigmas, const arma::vec &nus, const arma::field<arma::mat> &upsilons);

void aqqbnCorrVecs(arma::mat &phi, const arma::mat &x0, const arma::umat &z0, const arma::umat &w0, const arma::field<arma::mat> &v0,
                   const arma::mat &x, const arma::umat &z, const arma::umat &w, const arma::field<arma::mat> &v, 
                   const arma::mat &alpha, const arma::field<arma::mat> &taus, 
                   const arma::field<arma::mat> &thetas, const arma::field<arma::mat> &gammas,
                   const arma::vec &sigmas, const arma::vec &nus, const arma::field<arma::mat> &upsilons);

void aqqbnCorrVAR(arma::vec &predvar, const arma::umat &w0, 
                  const arma::vec &sigmas, const arma::vec &nus, const arma::field<arma::mat> &upsilons);

void aqqbnLogLik(double &negloglik, arma::mat &psi, arma::mat &invPsi, double &mu, 
                 const arma::vec &y, const arma::mat &x, const arma::umat &z, const arma::umat &w, const arma::field<arma::mat> &v, 
                 const arma::mat &alpha, const arma::field<arma::mat> &taus, 
                 const arma::field<arma::mat> &thetas, const arma::field<arma::mat> &gammas,
                 const arma::vec &sigmas, const arma::vec &nus, const arma::field<arma::mat> &upsilons);

void aqqbnNewData(arma::vec &y0, arma::vec &mse, arma::vec &ei, arma::vec &ei_1, arma::vec &ei_2, double &ei_alpha, double &min_y,
                  const arma::mat &x0, const arma::umat &z0, const arma::umat &w0, const arma::field<arma::mat> &v0,
                  const arma::vec &y, const arma::mat &x, const arma::umat &z, const arma::umat &w, const arma::field<arma::mat> &v, 
                  double &mu, arma::mat &invPsi, const arma::mat &alpha, const arma::field<arma::mat> &taus, 
                  const arma::field<arma::mat> &thetas, const arma::field<arma::mat> &gammas,
                  const arma::vec &sigmas, const arma::vec &nus, const arma::field<arma::mat> &upsilons);

void aqqbngpParam2vec(arma::mat &alpha, arma::field<arma::mat> &taus, 
                      arma::field<arma::mat> &thetas, arma::field<arma::mat> &gammas,
                      arma::vec &sigmas, arma::vec &nus, arma::field<arma::mat> &upsilons,
                      const arma::rowvec &param, const arma::uword &xDim, const arma::uword &zDim, const arma::uword &wDim, 
                      const arma::uvec &vDim, const arma::uvec &zlv, const arma::uvec &wlv, 
                      const arma::umat &z, const arma::umat &w, const arma::field<arma::mat> &v);

// BODY
// CORRELATION KERNEL OF GAUSSIAN PROCESS
double aqqbnCorrKern(const arma::rowvec &xi, const arma::urowvec &zi, const arma::urowvec &wi, const arma::field<arma::rowvec> &vi, 
                     const arma::rowvec &xj, const arma::urowvec &zj, const arma::urowvec &wj, const arma::field<arma::rowvec> &vj, 
                     const arma::mat &alpha, const arma::field<arma::mat> &taus, 
                     const arma::field<arma::mat> &thetas, const arma::field<arma::mat> &gammas,
                     const arma::vec &sigmas, const arma::vec &nus, const arma::field<arma::mat> &upsilons) 
{
  arma::uword zDim = zi.n_elem;
  arma::uword wDim = wi.n_elem;
  arma::rowvec xDiffSq = arma::pow(xi - xj, 2);
  double val = 0.0;
  for (arma::uword u = 0; u < zDim; u++) {
    arma::mat tau = taus(u, 0);
    val += sigmas(u)*tau(zi(u), zj(u))*std::exp(-(1.0)*arma::as_scalar(xDiffSq*alpha.col(u)));
  }
  for (arma::uword u = 0; u < wDim; u++) {
    arma::mat theta = thetas(u, 0);
    val += nus(u)*theta(wi(u), wj(u))*std::exp(-(1.0)*arma::as_scalar(xDiffSq*alpha.col(zDim + u)));
    if (wi(u) == wj(u)) {
      arma::rowvec v_u_i = vi(u, 0);
      arma::rowvec v_u_j = vj(u, 0);
      arma::uword ku = v_u_j.n_elem; // Dimension of v^(u)
      //gamma_u is a (n_v) x (n_lv) matrix
      arma::mat upsilon = upsilons(u, 0);
      arma::mat gamma_u = gammas(u, 0);
      arma::vec gamma_u_lv = gamma_u.col(wi(u));
      double rn = 0.0;
      for (arma::uword k = 0; k < ku; k++) {
        if (!std::isnan(v_u_i(k))) {
          rn += gamma_u_lv(k)*std::pow(v_u_i(k) - v_u_j(k), 2);
        }
      }
      val += upsilon(wi(u), 0)*std::exp(-(1.0)*rn);
    }
  }
  return val;
}


void aqqbnCorrMat(arma::mat &psi, const arma::mat &x, const arma::umat &z, const arma::umat &w, const arma::field<arma::mat> &v, 
                  const arma::mat &alpha, const arma::field<arma::mat> &taus, 
                  const arma::field<arma::mat> &thetas, const arma::field<arma::mat> &gammas,
                  const arma::vec &sigmas, const arma::vec &nus, const arma::field<arma::mat> &upsilons)
{
  arma::uword n = x.n_rows;
  arma::uword wDim = w.n_cols;
  for (uword i = 0; i < n; i++) {
    for (uword j = 0; j <= i; j++) {
      arma::rowvec xi = x.row(i);   arma::rowvec xj = x.row(j);
      arma::urowvec zi = z.row(i);  arma::urowvec zj = z.row(j);
      arma::urowvec wi = w.row(i);  arma::urowvec wj = w.row(j);
      arma::field<arma::rowvec> vi(wDim, 1); arma::field<arma::rowvec> vj(wDim, 1);
      for (uword u = 0; u < wDim; u++) {
        arma::mat vtmp = v(u, 0); vi(u, 0) = vtmp.row(i); vj(u, 0) = vtmp.row(j);
      }
      double ker = aqqbnCorrKern(xi, zi, wi, vi, xj, zj, wj, vj, alpha, taus, thetas, gammas, sigmas, nus, upsilons);
      psi(i, j) = ker;
      if (i != j) { psi(j, i) = ker; }
    }
  }
}

void aqqbnCorrVecs(arma::mat &phi, const arma::mat &x0, const arma::umat &z0, const arma::umat &w0, const arma::field<arma::mat> &v0,
                   const arma::mat &x, const arma::umat &z, const arma::umat &w, const arma::field<arma::mat> &v, 
                   const arma::mat &alpha, const arma::field<arma::mat> &taus, 
                   const arma::field<arma::mat> &thetas, const arma::field<arma::mat> &gammas,
                   const arma::vec &sigmas, const arma::vec &nus, const arma::field<arma::mat> &upsilons)
{
  arma::uword n = x.n_rows;
  arma::uword n0 = x0.n_rows;
  arma::uword wDim = w.n_cols;
  for (uword j = 0; j < n0; j++) {
    arma::rowvec x0j = x0.row(j); arma::urowvec z0j = z0.row(j); arma::urowvec w0j = w0.row(j);
    arma::field<arma::rowvec> v0j(wDim, 1);
    for (uword u = 0; u < wDim; u++) { arma::mat vtmp = v0(u, 0); v0j(u, 0) = vtmp.row(j); }
    for (uword i = 0; i < n; i++) {
      arma::rowvec xi = x.row(i); arma::urowvec zi = z.row(i); arma::urowvec wi = w.row(i);  
      arma::field<arma::rowvec> vi(wDim, 1); 
      for (uword u = 0; u < wDim; u++) { arma::mat vtmp = v(u, 0); vi(u, 0) = vtmp.row(i); }
      double ker = aqqbnCorrKern(xi, zi, wi, vi, x0j, z0j, w0j, v0j, alpha, taus, thetas, gammas, sigmas, nus, upsilons);
      phi(i, j) = ker;
    }
  }
}

void aqqbnCorrVAR(arma::vec &predvar, const arma::umat &w0, 
                  const arma::vec &sigmas, const arma::vec &nus, const arma::field<arma::mat> &upsilons)
{
  arma::uword n0 = w0.n_rows;
  arma::uword wDim = w0.n_cols;
  double common_v = arma::accu(sigmas) + arma::accu(nus);
  for (uword j = 0; j < n0; j++) {
    arma::urowvec w0j = w0.row(j);
    double branch_v = 0.0;
    for (arma::uword u = 0; u < wDim; u++) {
      arma::mat upsilon = upsilons(u, 0);
      branch_v += upsilon(w0j(u), 0);
    }
    predvar(j) = common_v + branch_v;
  }
}


void aqqbnLogLik(double &negloglik, arma::mat &psi, arma::mat &invPsi, double &mu, double &nugget,
                 const arma::vec &y, const arma::mat &x, const arma::umat &z, const arma::umat &w, const arma::field<arma::mat> &v, 
                 const arma::mat &alpha, const arma::field<arma::mat> &taus, 
                 const arma::field<arma::mat> &thetas, const arma::field<arma::mat> &gammas,
                 const arma::vec &sigmas, const arma::vec &nus, const arma::field<arma::mat> &upsilons)
{
  arma::uword n = y.n_elem;
  arma::vec onevec(n, fill::ones);
  aqqbnCorrMat(psi, x, z, w, v, alpha, taus, thetas, gammas, sigmas, nus, upsilons);
  arma::vec eigval;
  arma::mat eigvec;
  arma::eig_sym(eigval, eigvec, psi);
  arma::mat eyemat(n, n, fill::eye);
  double checkCond = std::abs(eigval.max()) - 1e8*std::abs(eigval.min());
  if ((nugget == 0) & (checkCond >= 0)) {
    nugget = checkCond/(1e8 - 1);
  }
  psi += nugget*eyemat;
  double detPsi;
  double signDetPsi;
  bool invSucc;
  invSucc = arma::inv_sympd(invPsi, psi);
  arma::log_det(detPsi, signDetPsi, psi);
  //if (std::isfinite(detPsi) & (signDetPsi >= 0)) 
  if (invSucc) {
    double yPsiY = arma::as_scalar(y.t()*invPsi*y);
    double onePsiY = arma::as_scalar(onevec.t()*invPsi*y);
    double onePsiOne = arma::as_scalar(onevec.t()*invPsi*onevec);
    mu = onePsiY/onePsiOne;
    negloglik = (-1.0)*(-0.5)*(detPsi + yPsiY - (onePsiY*onePsiY)/onePsiOne);
  } else {
    negloglik = 1e20;
  }
}

void aqqbnNewData(arma::vec &y0, arma::vec &mse, arma::vec &ei, arma::vec &ei_1, arma::vec &ei_2, double &ei_alpha, double &min_y,
                  const arma::mat &x0, const arma::umat &z0, const arma::umat &w0, const arma::field<arma::mat> &v0,
                  const arma::vec &y, const arma::mat &x, const arma::umat &z, const arma::umat &w, const arma::field<arma::mat> &v, 
                  double &mu, arma::mat &invPsi, const arma::mat &alpha, const arma::field<arma::mat> &taus, 
                  const arma::field<arma::mat> &thetas, const arma::field<arma::mat> &gammas,
                  const arma::vec &sigmas, const arma::vec &nus, const arma::field<arma::mat> &upsilons)
{
  arma::uword n = x.n_rows;
  arma::uword n0 = x0.n_rows;
  arma::mat phi(n, n0, fill::zeros);
  aqqbnCorrVecs(phi, x0, z0, w0, v0, x, z, w, v, alpha, taus, thetas, gammas, sigmas, nus, upsilons);
  arma::vec predvar(n0, fill::zeros);
  aqqbnCorrVAR(predvar, w0, sigmas, nus, upsilons);
  arma::vec onevec(n, fill::ones);
  arma::vec resid = y - mu*onevec;
  arma::vec psiinvresid = invPsi*resid;
  for (uword j = 0; j < n0; j++) {
    y0(j) = mu + arma::as_scalar(phi.col(j).t()*psiinvresid);
    mse(j) = std::abs(predvar(j) - arma::as_scalar(phi.col(j).t()*invPsi*phi.col(j))) + datum::eps;
  }
  // Compute expected improvement
  //double min_val = arma::min(y);
  arma::vec rmse = arma::sqrt(arma::abs(mse));
  arma::vec yd = min_y - y0;
  // The improvement part
  ei_1 = yd % (.5 + .5*arma::erf((1./std::sqrt(2.))*(yd/rmse)));
  // The uncertainty part
  ei_2 = (rmse/std::sqrt(2.*datum::pi)) % arma::exp(-.5*(yd % yd)/mse);
  // The EI value
  ei = 2.*(ei_alpha*ei_1 + (1. - ei_alpha)*ei_2);
  ei.elem( arma::find(ei <= .0) ).fill(datum::eps);
}


void aqqbngpParam2vec(arma::mat &alpha, arma::field<arma::mat> &taus, 
                      arma::field<arma::mat> &thetas, arma::field<arma::mat> &gammas,
                      arma::vec &sigmas, arma::vec &nus, arma::field<arma::mat> &upsilons,
                      const arma::rowvec &param, const arma::uword &xDim, const arma::uword &zDim, const arma::uword &wDim, 
                      const arma::uvec &vDim, const arma::uvec &zlv, const arma::uvec &wlv, 
                      const arma::umat &z, const arma::umat &w, const arma::field<arma::mat> &v)
{
  alpha.set_size(xDim, zDim + wDim); 
  taus.set_size(zDim, 1); 
  thetas.set_size(wDim, 1); 
  gammas.set_size(wDim, 1);
  sigmas.set_size(zDim);
  nus.set_size(wDim);
  upsilons.set_size(wDim, 1);
  // Conti #(alpha) = xDim*(zDim + wDim)
  // Categ #(taus) = SUM( choose(zlv(u), 2)) 
  // Var   #(sigmas) = zDim
  // Categ #(thetas) = SUM( choose(wlv(u), 2)) 
  // Var  #(nus) = wDim
  // Conti #(gammas) = SUM( vDim(u)*wlv(u) )
  // Var  #(upsilons) = SUM( wlv(u) )
  /* 
    START ASSIGN PARAMETER POSITION 
  */
  arma::uword ngammas = 0;
  arma::uword nupsilons = 0;
  for (arma::uword u = 0; u < wDim; u++) { ngammas += wlv(u)*vDim(u); nupsilons += wlv(u); }
  /* 
    Parameters for Continuous variables
  */
  // #(alpha) = xDim*(zDim + wDim)
  alpha = arma::reshape(param.subvec(0, xDim*(zDim + wDim) - 1), xDim, zDim + wDim);
  //
  arma::uword ct = xDim*(zDim + wDim);
  //
  // #(gammas) = SUM( vDim(u)*wlv(u) )
  arma::vec gammav = param.subvec(ct, ct + ngammas - 1).t(); // length = sum (n_v_u)*(n_wlv_u)
  arma::uword gct = 0;
  for (uword u = 0; u < wDim; u++) {
    arma::mat gtmp(vDim(u), wlv(u));
    /* gamma_u is a (n_v) x (n_wlv) matrix */
    for (arma::uword k = 0; k < wlv(u); k++) {
      gtmp.col(k) = gammav.subvec(gct, gct + vDim(u) - 1);
      gct += vDim(u);
    }
    gammas(u, 0) = gtmp;
  }
  ct += ngammas;
  /* 
    Parameters for Categorical variables
  */
  // #(taus) = SUM( choose(zlv(u), 2) ) 
  for (arma::uword u = 0; u < zDim; u++) {
    arma::mat tau(zlv(u), zlv(u), fill::eye);
    for (arma::uword i = 0; i < zlv(u); i++) {
      for (arma::uword j = 0; j < i; j++) {
        tau(i,j) = param(ct); tau(j,i) = param(ct); ct++;
      }
    }
    taus(u,0) = tau;
  }
  // #(thetas) = SUM( choose(wlv(u), 2) ) 
  for (arma::uword u = 0; u < wDim; u++) {
    arma::mat theta(wlv(u), wlv(u), fill::eye);
    for (arma::uword i = 0; i < wlv(u); i++) {
      for (arma::uword j = 0; j < i; j++) {
        theta(i,j) = param(ct); theta(j,i) = param(ct); ct++;
      }
    }
    thetas(u,0) = theta;
  }
  /* 
    Parameters for Variances
  */
  // #(sigmas) = zDim
  for (arma::uword u = 0; u < zDim; u++) { sigmas(u) = param(ct); ct++; }
  // #(nus) = wDim
  for (arma::uword u = 0; u < wDim; u++) { nus(u) = param(ct); ct++; }
  // #(upsilons) = SUM( wlv(u) )
  arma::vec upsilonv = param.subvec(ct, ct + nupsilons - 1).t(); // length = sum (n_wlv_u)
  arma::uword uct = 0;
  for (uword u = 0; u < wDim; u++) {
    upsilons(u, 0) = arma::reshape(upsilonv.subvec(uct, uct + wlv(u) - 1), wlv(u), 1);
    uct += wlv(u);
  }
}
