typedef int ssht_dl_size_t;
typedef int ssht_dl_method_t;

int ssht_sampling_mw_n(int L);
int ssht_sampling_mw_ntheta(int L);
int ssht_sampling_mw_nphi(int L);

double ssht_sampling_mw_t2theta(int t, int L);
double ssht_sampling_mw_p2phi(int p, int L);
double ssht_sampling_mw_ss_t2theta(int t, int L);
double ssht_sampling_mw_ss_p2phi(int p, int L);

void ssht_sampling_gl_thetas_weights(double *thetas, double *weights, int L);
double ssht_sampling_gl_p2phi(int p, int L);

void ssht_core_mw_forward_sov_conv_sym(double _Complex *flm,
                                       const double _Complex *f,
                        		       int L, int spin,
                        		       ssht_dl_method_t dl_method,
                        		       int verbosity);

void ssht_core_mw_inverse_sov_sym(double _Complex *f,
                                  const double _Complex *flm,
                                  int L, int spin,
                                  ssht_dl_method_t dl_method,
                                  int verbosity);

void ssht_core_mw_forward_sov_conv_sym_real(double _Complex *flm,
                                            const double *f,
                                            int L,
                                            ssht_dl_method_t dl_method,
                                            int verbosity);

void ssht_core_mw_inverse_sov_sym_real(double *f,
                                       const double _Complex *flm,
                                       int L,
                                       ssht_dl_method_t dl_method,
                                       int verbosity);

void ssht_core_mw_forward_sov_conv_sym_ss(double _Complex *flm,
                                          const double _Complex *f,
                                          int L, int spin,
                                          ssht_dl_method_t dl_method,
                                          int verbosity);

void ssht_core_mw_inverse_sov_sym_ss(double _Complex *f,
                                     const double _Complex *flm,
                                     int L, int spin,
                                     ssht_dl_method_t dl_method,
                                     int verbosity);

void ssht_core_mw_forward_sov_conv_sym_ss_real(double _Complex *flm,
                                               const double *f,
                                               int L,
                                               ssht_dl_method_t dl_method,
                                               int verbosity);

void ssht_core_mw_inverse_sov_sym_ss_real(double *f,
                                          const double _Complex *flm,
                                          int L,
                                          ssht_dl_method_t dl_method,
                                          int verbosity);

void ssht_core_gl_forward_sov(double _Complex *flm, const double _Complex *f,
                          int L, int spin, int verbosity);

void ssht_core_gl_inverse_sov(double _Complex *f, const double _Complex *flm,
                          int L, int spin, int verbosity);
