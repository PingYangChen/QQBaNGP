// DECLARE FUNCTIONS
void matrixPrintf(const mat &m);
void vecPrintf(const vec &v);
void rvecPrintf(const rowvec &v);

// BODY
void matrixPrintf(const mat &m)
{
  for (uword i = 0; i < m.n_rows; i++) {
    for (uword j = 0; j < m.n_cols; j++) Rprintf("%4.4f\t", m(i,j));
    Rprintf("\n");
  }
	Rprintf("\n\n");
}

void rvecPrintf(const rowvec &v)
{
  for (uword i = 0; i < v.n_elem; i++) Rprintf("%4.4f\t", v(i));
  Rprintf("\n\n");
}

void vecPrintf(const vec &v)
{
  for (uword i = 0; i < v.n_elem; i++) Rprintf("%4.4f\n", v(i));
  Rprintf("\n\n");
}
