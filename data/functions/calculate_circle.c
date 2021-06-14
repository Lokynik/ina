//#include <iostream>
#include "math.h"
#include "header.h"


void calculate_circle( int n, int n_start,int len_one_trace, double *t, double *v_c,
   double v_rev,double *v_cp,double *v_p,double *v_m,double *v_comp,
   double *m,double *h,double *j,
   double *I_leak, double *I_Na, double *args){
  //n, n_start, t, v_c, v_rev,v_cp,v_p,v_m,v_comp, m_inf,h_inf ,
  // m, h, j,I_leak, I_Na,args){

  double c_p = args[0];
  double c_m = args[1];
  double a0_m = args[2];
  double b0_m = args[3];
  double delta_m = args[4];
  double s_m = args[5];
  double a0_h = args[6];
  double b0_h = args[7];
  double delta_h = args[8];
  double s_h = args[9];
  double a0_j = args[10];
  double b0_j = args[11];
  double delta_j = args[12];
  double s_j = args[13];
  double tau_j_const = args[14];
  double r_m = args[15];
  double r_p = args[16];
  double g_max = args[17];
  double g_leak = args[18];
  double v_half_m = args[19];
  double v_half_h = args[20];
  double k_m = args[21];
  double k_h = args[22];
  double x_c_comp = args[23];
  double x_r_comp = args[24];
  double alpha = args[25];
  double v_off =  args[26];

  //int n_step = 25000;
  double dt = t[1] - t[0];
  //int time_ = 25000;

  I_leak[0] = g_leak * v_m[0];
  double c_comp = x_c_comp*c_m;
  double r_comp = x_r_comp*r_m;
  double tau_srp = r_m * c_m * (1 - alpha);
  double betta = 1/(1-alpha) - 1;


  double alpha_m = 0;
  double beta_m = 0;
  double alpha_h = 0;
  double beta_h = 0;
  double alpha_j = 0;
  double beta_j = 0;
  double tau_m = 0;
  double tau_h = 0;
  double tau_j = 0;
  double m_inf = 0;
  double h_inf = 1;


  int circle = 0;
  int i=1;
  while (i !=  n) {
      alpha_m  = a0_m * exp(v_m[i-1] / (s_m));
      beta_m = b0_m * exp(v_m[i-1] / (-delta_m));

      alpha_h  = a0_h * exp(v_m[i-1] / (-s_h));
      beta_h = b0_h * exp(v_m[i-1] / delta_h);

      alpha_j  = a0_j * exp(v_m[i-1] / (s_j)) ;
      beta_j = b0_j * exp(v_m[i-1] / (-delta_j));

      tau_m = 1 / (beta_m + alpha_m);//#+0.000037;
      tau_h = 1 / (beta_h + alpha_h);//+0.0002   ;
      tau_j = tau_j_const + 1 / (beta_j + alpha_j) ;

      v_comp[i] = v_c[i-1] + (v_comp[i-1] - v_c[i-1]) * exp(-dt / tau_srp);
      v_cp[i] =  v_c[i-1] + (v_c[i-1] - v_comp[i-1])*betta ;
      v_p[i] = v_cp[i-1] + (v_p[i-1] - v_cp[i-1]) * exp(-dt / (r_p * c_p));
      v_m[i] = v_m[i-1] + (v_p[i-1] + v_off - v_m[i-1] ) * (dt / (r_m * c_m)) - 1e-9 * (I_Na[i-1] + I_leak[i-1]) * dt / c_m ;

      m[i] = m_inf + (m[i-1] - m_inf) * exp(-dt/tau_m);
      h[i] = h_inf + (h[i-1] - h_inf) * exp(-dt/tau_h);
      j[i] = h_inf + (j[i-1] - h_inf) * exp(-dt/tau_j);

      m_inf = 1 / (1 + exp((- v_half_m - v_m[i]) / k_m));
      h_inf = 1 / (1 + exp((v_half_h + v_m[i]) / k_h));

      I_leak[i] = g_leak * v_m[i];
      I_Na[i] = g_max * h[i] * pow(m[i],3) * (v_m[i] - v_rev) * j[i] ;
      //if ((i-1)%n_start == 0 && circle != len_one_trace) {
      if (circle < n_start) {
          v_cp[i-1] = v_cp[i];
          v_p[i-1] = v_p[i];
          v_m[i-1] = v_m[i];
          v_comp[i-1] = v_comp[i];
          m[i-1] = m[i];
          h[i-1] = h[i];
          j[i-1] = j[i];
          I_leak[i-1] = I_leak[i];
          I_Na[i-1] = I_Na[i];
          circle ++;
          //#print(circle, i)
        }
      else {
          if (circle++ > len_one_trace+n_start) circle = 0;
          i += 1;
        }
     }
// v_cp,  v_p, v_m, v_comp, I_leak, I_Na;//tau_m[::n], tau_h[::n], tau_j[::n],

}
//extern "C" {
//  void calculate_circle( int n, int n_start, double t[25000], double v_c[25000],
//   double v_rev,double v_cp[25000],double v_p[25000],double v_m[25000],double v_comp[25000],
//   double m[25000],double h[25000],double j[25000],
//   double I_leak[25000], double I_Na[25000],double args[27]);
 //}

//#int main(){
//#/  int n = 10;
//#  int n_start = 15e3;
//#  t =
//#  double v_c, v_rev,v_cp,v_p,v_m,v_comp, m_inf,h_inf , m, h, j,I_leak, I_Na,


//#  double p0 [27] = {6.00e-13, 5.00e-11, //#C_p , C
//#/               2.40e+10, 6.90e+01, 1.2e-01, 2.86e+03, //#a0_m , b0_m , delta_m , s_m
//               1.15e-01, 1.57e+03, 2.4e+01,  4.6e+01, //#a0_h , b0_h , delta_h , s_h
  //             3.13e+03, 1.13e-04, 8.28e+00, 3.84e+01, 2.00e-02,//#a0_j , b0_j , delta_j , s_j , tau_j_const
///               2.50e+07, 7.00e+04, 4.27e+02, 1.00e+0,//#R , R_p , g_max , g_leak
//               1.04e+01, 4.64e+01, 1.30e+01, 2.80e-01,//#v_half_m , v_half_h , k_m, k_h
//               8.00e-01, 7.90e-01, 0.10e-03, 0.00e+00}
//  calculate_circle(n, n_start, t, v_c, v_rev,v_cp,v_p,v_m,v_comp, m_inf,h_inf , m, h, j,I_leak, I_Na,p0)
//}
