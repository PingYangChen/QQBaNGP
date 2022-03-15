// HEADER

double qqbnCorrKern(const arma::rowvec &xi, const arma::urowvec &zi, const arma::urowvec &wi, const arma::field<arma::rowvec> &vi, 
                    const arma::rowvec &xj, const arma::urowvec &zj, const arma::urowvec &wj, const arma::field<arma::rowvec> &vj, 
                    const arma::vec &alpha, const arma::vec &tau, const arma::vec &theta, const arma::field<arma::mat> &gammas);

void qqbnCorrMat(arma::mat &psi, const arma::mat &x, const arma::umat &z, const arma::umat &w, const arma::field<arma::mat> &v, 
                 const arma::vec &alpha, const arma::vec &tau, const arma::vec &theta, const arma::field<arma::mat> &gammas);

void qqbnCorrVecs(arma::mat &phi, const arma::mat &x0, const arma::umat &z0, const arma::umat &w0, const arma::field<arma::mat> &v0,
                  const arma::mat &x, const arma::umat &z, const arma::umat &w, const arma::field<arma::mat> &v, 
                  const arma::vec &alpha, const arma::vec &tau, const arma::vec &theta, const arma::field<arma::mat> &gammas);

void qqbnLogLik(double &negloglik, arma::mat &psi, arma::mat &invPsi, double &mu, double &sigma,  double &nugget,
                const arma::vec &y, const arma::mat &x, const arma::umat &z, const arma::umat &w, const arma::field<arma::mat> &v, 
                const arma::vec &alpha, const arma::vec &tau, const arma::vec &theta, const arma::field<arma::mat> &gammas);

void qqbnNewData(arma::vec &y0, arma::vec &mse, arma::vec &ei, arma::vec &ei_1, arma::vec &ei_2, double &ei_alpha, double &min_y,
                 const arma::mat &x0, const arma::umat &z0, const arma::umat &w0, const arma::field<arma::mat> &v0,
                 const arma::vec &y, const arma::mat &x, const arma::umat &z, const arma::umat &w, const arma::field<arma::mat> &v, 
                 double &mu, double &sigma, arma::mat &invPsi, 
                 const arma::mat &alpha, const arma::vec &tau, const arma::vec &theta, const arma::field<arma::mat> &gammas);

void qqbngpParam2vec(arma::vec &alpha, arma::vec &tau, arma::vec &theta, arma::field<arma::mat> &gammas, 
                     const arma::rowvec &param, const arma::uword &xDim, const arma::uword &zDim, const arma::uword &wDim, 
                     const arma::uvec &vDim, const arma::uvec &zlv, const arma::uvec &wlv, 
                     const arma::umat &z, const arma::umat &w, const arma::field<arma::mat> &v);

// BODY
// CORRELATION KERNEL (EXCHANGEABLE)
double qqbnCorrKern(const arma::rowvec &xi, const arma::urowvec &zi, const arma::urowvec &wi, const arma::field<arma::rowvec> &vi, 
                    const arma::rowvec &xj, const arma::urowvec &zj, const arma::urowvec &wj, const arma::field<arma::rowvec> &vj, 
                    const arma::vec &alpha, const arma::vec &tau, const arma::vec &theta, const arma::field<arma::mat> &gammas) 
{
  arma::uword zDim = zi.n_elem;
  arma::uword wDim = wi.n_elem;
  arma::rowvec xDiffSq = arma::pow(xi - xj, 2);
  double rx = arma::as_scalar(xDiffSq*alpha);
  double rz = 0.0;
  for (uword j = 0; j < zDim; j++) {
    if (zi(j) != zj(j)) {
      rz += tau(j);
    }
  }
  double rbn = 0.0;
  for (uword u = 0; u < wDim; u++) {
    if (wi(u) == wj(u)) {
      arma::rowvec v_u_i = vi(u, 0);
      arma::rowvec v_u_j = vj(u, 0);
      arma::uword ku = v_u_j.n_elem;
      /*
      gamma_u is a (n_v) x (n_lv) matrix
      */
      arma::mat gamma_u = gammas(u, 0);
      arma::vec gamma_u_lv = gamma_u.col(wi(u));
      for (arma::uword k = 0; k < ku; k++) {
        if (!std::isnan(v_u_i(k))) {
          rbn += gamma_u_lv(k)*std::pow(v_u_i(k) - v_u_j(k), 2);
        }
      }
    } else {
      rbn += theta(u);
    }
  }
  double val = std::exp(-(1.0)*(rx + rz + rbn));
  return val;
}


void qqbnCorrMat(arma::mat &psi, const arma::mat &x, const arma::umat &z, const arma::umat &w, const arma::field<arma::mat> &v, 
                 const arma::vec &alpha, const arma::vec &tau, const arma::vec &theta, const arma::field<arma::mat> &gammas)
{
  arma::uword n = x.n_rows;
  arma::uword wDim = w.n_cols;
  for (uword i = 0; i < n; i++) {
    for (uword j = 0; j < i; j++) {
      arma::rowvec xi = x.row(i);   arma::rowvec xj = x.row(j);
      arma::urowvec zi = z.row(i);  arma::urowvec zj = z.row(j);
      arma::urowvec wi = w.row(i);  arma::urowvec wj = w.row(j);
      arma::field<arma::rowvec> vi(wDim, 1); arma::field<arma::rowvec> vj(wDim, 1);
      for (uword u = 0; u < wDim; u++) {
        arma::mat vtmp = v(u, 0); vi(u, 0) = vtmp.row(i); vj(u, 0) = vtmp.row(j);
      }
      double ker = qqbnCorrKern(xi, zi, wi, vi, xj, zj, wj, vj, alpha, tau, theta, gammas);
      psi(i, j) = ker;
      psi(j, i) = ker;
    }
  }
}

void qqbnCorrVecs(arma::mat &phi, const arma::mat &x0, const arma::umat &z0, const arma::umat &w0, const arma::field<arma::mat> &v0,
                  const arma::mat &x, const arma::umat &z, const arma::umat &w, const arma::field<arma::mat> &v, 
                  const arma::vec &alpha, const arma::vec &tau, const arma::vec &theta, const arma::field<arma::mat> &gammas)
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
      double ker = qqbnCorrKern(xi, zi, wi, vi, x0j, z0j, w0j, v0j, alpha, tau, theta, gammas);
      phi(i, j) = ker;
    }
  }
}

void qqbnLogLik(double &negloglik, arma::mat &psi, arma::mat &invPsi, double &mu, double &sigma, double &nugget,
                const arma::vec &y, const arma::mat &x, const arma::umat &z, const arma::umat &w, const arma::field<arma::mat> &v, 
                const arma::vec &alpha, const arma::vec &tau, const arma::vec &theta, const arma::field<arma::mat> &gammas)
{
  arma::uword n = y.n_elem;
  double n_double = (double)n;
  arma::vec onevec(n, fill::ones);
  qqbnCorrMat(psi, x, z, w, v, alpha, tau, theta, gammas);
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
    mu = arma::as_scalar(onevec.t()*invPsi*y)/arma::as_scalar(onevec.t()*invPsi*onevec);
    arma::vec res = y - (mu*onevec);
    sigma = arma::as_scalar(res.t()*invPsi*res)/n_double;
    negloglik = .5*(n_double*std::log(sigma + datum::eps) + detPsi); 
  } else {
    negloglik = 1e20;
  }
}

void qqbnNewData(arma::vec &y0, arma::vec &mse, arma::vec &ei, arma::vec &ei_1, arma::vec &ei_2, double &ei_alpha, double &min_y,
                 const arma::mat &x0, const arma::umat &z0, const arma::umat &w0, const arma::field<arma::mat> &v0,
                 const arma::vec &y, const arma::mat &x, const arma::umat &z, const arma::umat &w, const arma::field<arma::mat> &v, 
                 double &mu, double &sigma, arma::mat &invPsi, 
                 const arma::mat &alpha, const arma::vec &tau, const arma::vec &theta, const arma::field<arma::mat> &gammas)
{
  arma::uword n = x.n_rows;
  arma::uword n0 = x0.n_rows;
  arma::mat phi(n, n0, fill::zeros);
  qqbnCorrVecs(phi, x0, z0, w0, v0, x, z, w, v, alpha, tau, theta, gammas);
  arma::vec onevec(n, fill::ones);
  arma::vec resid = y - mu*onevec; 
  arma::vec psiinvresid = invPsi*resid;
  for (uword j = 0; j < n0; j++) {
    y0(j) = mu + arma::as_scalar(phi.col(j).t()*psiinvresid);
    mse(j) = std::abs(sigma*(1. - arma::as_scalar(phi.col(j).t()*invPsi*phi.col(j)))) + datum::eps;
  }
  // Compute expected improvement
  //double min_val = arma::min(y);
  arma::vec rmse = arma::sqrt(mse);
  arma::vec yd = min_y - y0;
  // The improvement part
  ei_1 = yd % (.5 + .5*arma::erf((1./std::sqrt(2.))*(yd/rmse)));
  // The uncertainty part
  ei_2 = (rmse/std::sqrt(2.*datum::pi)) % arma::exp(-.5*(yd % yd)/mse);
  // The EI value
  ei = 2.*(ei_alpha*ei_1 + (1. - ei_alpha)*ei_2);
  ei.elem( arma::find(ei <= .0) ).fill(datum::eps);  
}


void qqbngpParam2vec(arma::vec &alpha, arma::vec &tau, arma::vec &theta, arma::field<arma::mat> &gammas, 
                     const arma::rowvec &param, const arma::uword &xDim, const arma::uword &zDim, const arma::uword &wDim, 
                     const arma::uvec &vDim, const arma::uvec &zlv, const arma::uvec &wlv, 
                     const arma::umat &z, const arma::umat &w, const arma::field<arma::mat> &v)
{
  alpha.set_size(xDim); theta.set_size(wDim); gammas.set_size(wDim, 1);
  arma::uword ngammas = 0;
  for (arma::uword u = 0; u < wDim; u++) { ngammas += wlv(u)*vDim(u); }
  /* 
    Parameters for Continuous variables
  */
  // #(alpha) = xDim
  alpha = param.subvec(0, xDim - 1).t();
  //
  arma::uword ct = xDim;
  // #(gammas) = SUM( vDim(u)*wlv(u) )
  arma::vec gammav = param.subvec(ct, ct + ngammas - 1).t(); 
  arma::uword g0 = 0;
  for (uword u = 0; u < wDim; u++) {
    arma::uword nlv = arma::max(w.col(u)) + 1;
    arma::uword vDim = v(u, 0).n_cols;
    arma::mat gtmp(vDim, nlv);
    /* gamma_u is a (n_v) x (n_lv) matrix */
    for (arma::uword k = 0; k < nlv; k++) {
      gtmp.col(k) = gammav.subvec(g0, g0 + vDim - 1);
      g0 += vDim;
    }
    gammas(u, 0) = gtmp;
  }
  ct += ngammas;
  /* 
    Parameters for Categorical variables
  */
  // #(taus) = zDim
  tau = param.subvec(ct, ct + zDim - 1).t();;
  ct += zDim;
  // #(thetas) = wDim
  theta = param.subvec(ct, ct + wDim - 1).t();
}

