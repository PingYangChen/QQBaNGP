
#include "header.h"

// RCPP FUNCTIONS

//[[Rcpp::export]]
double gpObjCpp(arma::vec param, arma::vec y, arma::mat x, double nugget)
{
  arma::uword n = x.n_rows;
  double negloglik, mu, sigma;
  arma::mat psi(n, n, fill::eye);
  arma::mat invPsi(n, n, fill::eye);
  //double nugget = 0.;
  gpLogLik(negloglik, psi, invPsi, mu, sigma, nugget, y, x, param);
  return negloglik;
}

//[[Rcpp::export]]
Rcpp::List gpModel(arma::vec param, arma::vec y, arma::mat x, double nugget)
{
  arma::uword n = x.n_rows;
  double negloglik, mu, sigma;
  arma::mat psi(n, n, fill::eye);
  arma::mat invPsi(n, n, fill::eye);
  //double nugget = 0.;
  gpLogLik(negloglik, psi, invPsi, mu, sigma, nugget, y, x, param);
  return List::create(Named("alpha") = wrap(param),
                      Named("psi") = wrap(psi),
                      Named("invPsi") = wrap(invPsi),
                      Named("mu") = wrap(mu),
                      Named("sigma") = wrap(sigma),
                      Named("negloglik") = wrap(negloglik),
                      Named("nugget") = wrap(nugget));
}

//[[Rcpp::export]]
Rcpp::List gpPred(arma::mat x0, arma::vec y, arma::mat x, 
                  arma::vec param, arma::mat invPsi, double mu, double sigma, double ei_alpha, double min_y)
{
  arma::uword n0 = x0.n_rows;
  arma::vec y0(n0, fill::zeros);
  arma::vec mse(n0, fill::zeros);
  arma::vec ei(n0, fill::zeros);  
  arma::vec ei_1(n0, fill::zeros);
  arma::vec ei_2(n0, fill::zeros);
  gpNewData(y0, mse, ei, ei_1, ei_2, ei_alpha, min_y, x0, y, x, mu, sigma, invPsi, param);
  //
  return List::create(Named("pred") = wrap(y0),
                      Named("mse") = wrap(mse),
                      Named("ei") = wrap(ei),
                      Named("improvement") = wrap(ei_1),
                      Named("uncertainty") = wrap(ei_2)
                     );
}


/*
   BNGP
*/
//[[Rcpp::export]]
double bngpObjCpp(arma::rowvec param, arma::vec y, arma::mat x, arma::umat w, Rcpp::List vList, double nugget)
{
  arma::field<arma::mat> v = list2field(vList);
  arma::uword n = x.n_rows;
  double negloglik, mu, sigma;
  arma::mat psi(n, n, fill::eye);
  arma::mat invPsi(n, n, fill::eye);
  arma::uword xDim = x.n_cols;
  arma::uword wDim = w.n_cols;
  //
  arma::uvec vDim(wDim);
  arma::uvec wlv(wDim);
  for (arma::uword u = 0; u < wDim; u++) { 
    wlv(u) = arma::max(w.col(u)) + 1; vDim(u) = v(u, 0).n_cols;
  }
  //
  arma::vec alpha, theta;
  arma::field<arma::mat> gammas;
  //double nugget = 0.;
  bngpParam2vec(alpha, theta, gammas, param, xDim, wDim, vDim, wlv, w, v);
  bnLogLik(negloglik, psi, invPsi, mu, sigma, nugget, y, x, w, v, alpha, theta, gammas);
  return negloglik;
}

//[[Rcpp::export]]
Rcpp::List bngpModel(arma::rowvec param, arma::vec y, arma::mat x, arma::umat w, Rcpp::List vList, double nugget)
{
  arma::field<arma::mat> v = list2field(vList);
  arma::uword n = x.n_rows;
  double negloglik, mu, sigma;
  arma::mat psi(n, n, fill::eye);
  arma::mat invPsi(n, n, fill::eye);
  arma::uword xDim = x.n_cols;
  arma::uword wDim = w.n_cols;
  //
  arma::uvec vDim(wDim);
  arma::uvec wlv(wDim);
  for (arma::uword u = 0; u < wDim; u++) { 
    wlv(u) = arma::max(w.col(u)) + 1; vDim(u) = v(u, 0).n_cols;
  }
  //
  arma::vec alpha, theta;
  arma::field<arma::mat> gammas;
  //double nugget = 0.;
  bngpParam2vec(alpha, theta, gammas, param, xDim, wDim, vDim, wlv, w, v);
  bnLogLik(negloglik, psi, invPsi, mu, sigma, nugget, y, x, w, v, alpha, theta, gammas);
  Rcpp::List gammalist = field2list(gammas);
  return List::create(Named("alpha") = wrap(alpha),
                      Named("theta") = wrap(theta),
                      Named("gamma") = wrap(gammalist),
                      Named("psi") = wrap(psi),
                      Named("invPsi") = wrap(invPsi),
                      Named("mu") = wrap(mu),
                      Named("sigma") = wrap(sigma),
                      Named("negloglik") = wrap(negloglik),
                      Named("nugget") = wrap(nugget),
                      Named("vecParams") = wrap(param));
}

//[[Rcpp::export]]
Rcpp::List bngpPred(arma::mat x0, arma::umat w0, Rcpp::List v0List,
                    arma::vec y, arma::mat x, arma::umat w, Rcpp::List vList,
                    arma::rowvec param, arma::mat invPsi, double mu, double sigma, double ei_alpha, double min_y)
{
  arma::uword n0 = x0.n_rows;
  arma::field<arma::mat> v = list2field(vList);
  arma::field<arma::mat> v0 = list2field(v0List);
  arma::uword xDim = x.n_cols;
  arma::uword wDim = w.n_cols;
  //
  arma::uvec vDim(wDim);
  arma::uvec wlv(wDim);
  for (arma::uword u = 0; u < wDim; u++) { 
    wlv(u) = arma::max(w.col(u)) + 1; vDim(u) = v(u, 0).n_cols;
  }
  //
  arma::vec alpha, theta;
  arma::field<arma::mat> gammas;
  bngpParam2vec(alpha, theta, gammas, param, xDim, wDim, vDim, wlv, w, v);
  arma::vec y0(n0, fill::zeros);
  arma::vec mse(n0, fill::zeros);
  arma::vec ei(n0, fill::zeros);  
  arma::vec ei_1(n0, fill::zeros);
  arma::vec ei_2(n0, fill::zeros);
  bnNewData(y0, mse, ei, ei_1, ei_2, ei_alpha, min_y, x0, w0, v0, y, x, w, v, mu, sigma, invPsi, alpha, theta, gammas);
  //
  return List::create(Named("pred") = wrap(y0),
                      Named("mse") = wrap(mse),
                      Named("ei") = wrap(ei),
                      Named("improvement") = wrap(ei_1),
                      Named("uncertainty") = wrap(ei_2)
                     );
}

/*
   QQBNGP
*/
//[[Rcpp::export]]
double qqbngpObjCpp(arma::rowvec param, arma::vec y, arma::mat x, arma::umat z, arma::umat w, Rcpp::List vList, double nugget)
{
  arma::field<arma::mat> v = list2field(vList);
  arma::uword n = x.n_rows;
  double negloglik, mu, sigma;
  arma::mat psi(n, n, fill::eye);
  arma::mat invPsi(n, n, fill::eye);
  arma::uword xDim = x.n_cols;
  arma::uword zDim = z.n_cols;
  arma::uword wDim = w.n_cols;
  //
  arma::uvec vDim(wDim);
  arma::uvec zlv(zDim);
  arma::uvec wlv(wDim);
  for (arma::uword u = 0; u < zDim; u++) { zlv(u) = arma::max(z.col(u)) + 1; }
  for (arma::uword u = 0; u < wDim; u++) { 
    wlv(u) = arma::max(w.col(u)) + 1; vDim(u) = v(u, 0).n_cols;
  }
  //
  arma::vec alpha, tau, theta;
  arma::field<arma::mat> gammas;
  //double nugget = 0.;
  qqbngpParam2vec(alpha, tau, theta, gammas, param, xDim, zDim, wDim, vDim, zlv, wlv, z, w, v);
  qqbnLogLik(negloglik, psi, invPsi, mu, sigma, nugget, y, x, z, w, v, alpha, tau, theta, gammas);
  return negloglik;
}

//[[Rcpp::export]]
Rcpp::List qqbngpModel(arma::rowvec param, arma::vec y, arma::mat x, arma::umat z, arma::umat w, Rcpp::List vList, double nugget)
{
  arma::field<arma::mat> v = list2field(vList);
  arma::uword n = x.n_rows;
  double negloglik, mu, sigma;
  arma::mat psi(n, n, fill::eye);
  arma::mat invPsi(n, n, fill::eye);
  arma::uword xDim = x.n_cols;
  arma::uword zDim = z.n_cols;
  arma::uword wDim = w.n_cols;
  //
  arma::uvec vDim(wDim);
  arma::uvec zlv(zDim);
  arma::uvec wlv(wDim);
  for (arma::uword u = 0; u < zDim; u++) { zlv(u) = arma::max(z.col(u)) + 1; }
  for (arma::uword u = 0; u < wDim; u++) { 
    wlv(u) = arma::max(w.col(u)) + 1; vDim(u) = v(u, 0).n_cols;
  }
  //
  arma::vec alpha, tau, theta;
  arma::field<arma::mat> gammas;
  //double nugget = 0.;
  qqbngpParam2vec(alpha, tau, theta, gammas, param, xDim, zDim, wDim, vDim, zlv, wlv, z, w, v);
  qqbnLogLik(negloglik, psi, invPsi, mu, sigma, nugget, y, x, z, w, v, alpha, tau, theta, gammas);
  Rcpp::List gammalist = field2list(gammas);
  return List::create(Named("alpha") = wrap(alpha),
                      Named("tau") = wrap(tau),
                      Named("theta") = wrap(theta),
                      Named("gamma") = wrap(gammalist),
                      Named("psi") = wrap(psi),
                      Named("invPsi") = wrap(invPsi),
                      Named("mu") = wrap(mu),
                      Named("sigma") = wrap(sigma),
                      Named("negloglik") = wrap(negloglik),
                      Named("nugget") = wrap(nugget),
                      Named("vecParams") = wrap(param));
}


//[[Rcpp::export]]
Rcpp::List qqbngpPred(arma::mat x0, arma::umat z0, arma::umat w0, Rcpp::List v0List,
                      arma::vec y, arma::mat x, arma::umat z, arma::umat w, Rcpp::List vList,
                      arma::rowvec param, arma::mat invPsi, double mu, double sigma, double ei_alpha, double min_y)
{
  arma::uword n0 = x0.n_rows;
  arma::field<arma::mat> v = list2field(vList);
  arma::field<arma::mat> v0 = list2field(v0List);
  arma::uword xDim = x.n_cols;
  arma::uword zDim = z.n_cols;
  arma::uword wDim = w.n_cols;
  //
  arma::uvec vDim(wDim);
  arma::uvec zlv(zDim);
  arma::uvec wlv(wDim);
  for (arma::uword u = 0; u < zDim; u++) { zlv(u) = arma::max(z.col(u)) + 1; }
  for (arma::uword u = 0; u < wDim; u++) { 
    wlv(u) = arma::max(w.col(u)) + 1; vDim(u) = v(u, 0).n_cols;
  }
  //
  arma::vec alpha, tau, theta;
  arma::field<arma::mat> gammas;
  qqbngpParam2vec(alpha, tau, theta, gammas, 
                  param, xDim, zDim, wDim, vDim, zlv, wlv, z, w, v);
  arma::vec y0(n0, fill::zeros);
  arma::vec mse(n0, fill::zeros);
  arma::vec ei(n0, fill::zeros);
  arma::vec ei_1(n0, fill::zeros);
  arma::vec ei_2(n0, fill::zeros);
  qqbnNewData(y0, mse, ei, ei_1, ei_2, ei_alpha, min_y, x0, z0, w0, v0, y, x, z, w, v, mu, sigma, invPsi, alpha, tau, theta, gammas);
  //
  return List::create(Named("pred") = wrap(y0),
                      Named("mse") = wrap(mse),
                      Named("ei") = wrap(ei),
                      Named("improvement") = wrap(ei_1),
                      Named("uncertainty") = wrap(ei_2)
                     );
}


/*
   AQQBNGP
*/
//[[Rcpp::export]]
double aqqbngpObjCpp(arma::rowvec param, arma::vec y, arma::mat x, arma::umat z, arma::umat w, Rcpp::List vList, double nugget)
{
  arma::field<arma::mat> v = list2field(vList);
  arma::uword n = x.n_rows;
  double negloglik, mu;
  arma::mat psi(n, n, fill::eye); 
  arma::mat invPsi(n, n, fill::eye);
  arma::uword xDim = x.n_cols;
  arma::uword zDim = z.n_cols;
  arma::uword wDim = w.n_cols;
  //
  arma::uvec vDim(wDim);
  arma::uvec zlv(zDim);
  arma::uvec wlv(wDim);
  for (arma::uword u = 0; u < zDim; u++) { zlv(u) = arma::max(z.col(u)) + 1; }
  for (arma::uword u = 0; u < wDim; u++) { 
    wlv(u) = arma::max(w.col(u)) + 1; vDim(u) = v(u, 0).n_cols;
  }
  //
  arma::mat alpha;
  arma::field<arma::mat> taus, thetas, gammas, upsilons;
  arma::vec sigmas, nus;
  //double nugget = 0.;
  aqqbngpParam2vec(alpha, taus, thetas, gammas, sigmas, nus, upsilons,
                   param, xDim, zDim, wDim, vDim, zlv, wlv, z, w, v);
  aqqbnLogLik(negloglik, psi, invPsi, mu, nugget, y, x, z, w, v, 
              alpha, taus, thetas, gammas, sigmas, nus, upsilons);
  return negloglik;
}

//[[Rcpp::export]]
Rcpp::List aqqbngpModel(arma::rowvec param, arma::vec y, arma::mat x, arma::umat z, arma::umat w, Rcpp::List vList, double nugget)
{
  arma::field<arma::mat> v = list2field(vList);
  arma::uword n = x.n_rows;
  double negloglik, mu;
  arma::mat psi(n, n, fill::eye); 
  arma::mat invPsi(n, n, fill::eye);
  arma::uword xDim = x.n_cols;
  arma::uword zDim = z.n_cols;
  arma::uword wDim = w.n_cols;
  //
  arma::uvec vDim(wDim);
  arma::uvec zlv(zDim);
  arma::uvec wlv(wDim);
  for (arma::uword u = 0; u < zDim; u++) { zlv(u) = arma::max(z.col(u)) + 1; }
  for (arma::uword u = 0; u < wDim; u++) { 
    wlv(u) = arma::max(w.col(u)) + 1; vDim(u) = v(u, 0).n_cols;
  }
  //
  arma::mat alpha;
  arma::field<arma::mat> taus, thetas, gammas, upsilons;
  arma::vec sigmas, nus;
  //double nugget = 0.;
  aqqbngpParam2vec(alpha, taus, thetas, gammas, sigmas, nus, upsilons,
                   param, xDim, zDim, wDim, vDim, zlv, wlv, z, w, v);
  aqqbnLogLik(negloglik, psi, invPsi, mu, nugget, y, x, z, w, v, 
              alpha, taus, thetas, gammas, sigmas, nus, upsilons);
  //
  Rcpp::List taulist = field2list(taus);
  Rcpp::List thetalist = field2list(thetas);
  Rcpp::List gammalist = field2list(gammas);
  Rcpp::List upsilonlist = field2list(upsilons);
  return List::create(Named("mu") = wrap(mu),
                      Named("alpha") = wrap(alpha),
                      Named("tau") = wrap(taulist),
                      Named("theta") = wrap(thetalist),
                      Named("gamma") = wrap(gammalist),
                      Named("sigma") = wrap(sigmas),
                      Named("nu") = wrap(nus),
                      Named("upsilon") = wrap(upsilonlist),
                      Named("psi") = wrap(psi),
                      Named("invPsi") = wrap(invPsi),
                      Named("negloglik") = wrap(negloglik),
                      Named("nugget") = wrap(nugget),
                      Named("vecParams") = wrap(param)
                     );
}

//[[Rcpp::export]]
Rcpp::List aqqbngpPred(arma::mat x0, arma::umat z0, arma::umat w0, Rcpp::List v0List,
                       arma::vec y, arma::mat x, arma::umat z, arma::umat w, Rcpp::List vList,
                       arma::rowvec param, arma::mat invPsi, double mu, double ei_alpha, double min_y)
{
  arma::uword n0 = x0.n_rows;
  arma::field<arma::mat> v = list2field(vList);
  arma::field<arma::mat> v0 = list2field(v0List);
  arma::uword xDim = x.n_cols;
  arma::uword zDim = z.n_cols;
  arma::uword wDim = w.n_cols;
  //
  arma::uvec vDim(wDim);
  arma::uvec zlv(zDim);
  arma::uvec wlv(wDim);
  for (arma::uword u = 0; u < zDim; u++) { zlv(u) = arma::max(z.col(u)) + 1; }
  for (arma::uword u = 0; u < wDim; u++) { 
    wlv(u) = arma::max(w.col(u)) + 1; vDim(u) = v(u, 0).n_cols;
  }
  //
  arma::mat alpha;
  arma::field<arma::mat> taus, thetas, gammas, upsilons;
  arma::vec sigmas, nus;
  aqqbngpParam2vec(alpha, taus, thetas, gammas, sigmas, nus, upsilons,
                   param, xDim, zDim, wDim, vDim, zlv, wlv, z, w, v);
  arma::vec y0(n0, fill::zeros);
  arma::vec mse(n0, fill::zeros);
  arma::vec ei(n0, fill::zeros);
  arma::vec ei_1(n0, fill::zeros);
  arma::vec ei_2(n0, fill::zeros);
  aqqbnNewData(y0, mse, ei, ei_1, ei_2, ei_alpha, min_y, x0, z0, w0, v0, y, x, z, w, v, mu, invPsi, alpha, taus, thetas, gammas,
               sigmas, nus, upsilons);
  //
  return List::create(Named("pred") = wrap(y0),
                      Named("mse") = wrap(mse),
                      Named("ei") = wrap(ei),
                      Named("improvement") = wrap(ei_1),
                      Named("uncertainty") = wrap(ei_2)
                     );
}









/*
//[[Rcpp::export]]
arma::vec qqbntsgpMuStep(arma::vec y, arma::mat x, arma::umat z, arma::umat w, Rcpp::List vList, 
                         arma::umat uniZW, arma::vec phi, Rcpp::List gammaList)
{
  arma::field<arma::mat> v = list2field(vList);
  arma::field<arma::mat> gammas = list2field(gammaList);
  arma::uword nLvComb = uniZW.n_rows;
  arma::vec mu(nLvComb, fill::zeros);
  muStep(mu, y, x, z, w, v, uniZW, phi, gammas);
  return mu;
}
*/

