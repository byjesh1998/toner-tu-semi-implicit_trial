#include <boost/iostreams/tee.hpp>
#include <boost/iostreams/stream.hpp>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <fstream>
#include "mkl.h"
#include <armadillo> //
#include <ctime>
#include <csignal>

using namespace std;
using namespace arma;  /// first commit test
//hello

extern "C" void create_descriptor_handles(int, int);
extern "C" void fft_forward(double *, MKL_Complex16 *);
extern "C" void fft_backward(MKL_Complex16 *in, double *out);
extern "C" void fft_padded_forward(double *, MKL_Complex16 *);
extern "C" void fft_padded_backward(MKL_Complex16 *in, double *out);
extern "C" void free_descriptor_handles();

void sample_noise(cx_double *z);
bool rdata(const string &input_dir);

// Convolution object
class Convolution {
public:
    Convolution();
    void convolve2d(Mat<cx_double> &out, Mat<cx_double> &A, Mat<cx_double> &B);
    void convolve2d_same(Mat<cx_double> &out, Mat<cx_double> &A);
    void pad_matrix(cx_double *out, cx_double *in);
    void remove_pad(cx_double *out, cx_double *in);
    void cleanup();

private:
    mat conv_out_1;
    mat conv_out_2;
    cx_mat conv_out_ft;
    cx_mat pad_ft1;
    cx_mat pad_ft2;
};

// SIGINT handler
volatile sig_atomic_t sigint_flag = 0;//
void sigint_handler(int sig) {
    sigint_flag = 1;
}

typedef boost::iostreams::tee_device<ostream, ofstream> TeeDevice;
typedef boost::iostreams::stream<TeeDevice> TeeStream;

// Solver properties
int steps;
int print_interval;
int Nx;
int Ny;
double Lx;
double Ly;
double dt;

// Parameters
double Diff;
double l;
double k;
double w;
double w1;

// Initial conditions
double rho0;
double Px0;
double Py0;

int main(int argc, const char **argv) {
    clock_t start;
    typedef std::numeric_limits<double> dbl;
    double duration;
    double sqrt_Ddt;
    double epr_av = 0;
    int average_shift = 0;
    bool continue_average = false;
    bool average_epr = false;
    bool verbose = false;
    bool no_noise = false;
    bool axy = false;
    string input_dir;
    char option;
    ofstream Px_file;
    ofstream Py_file;
    ofstream P_file;
    ofstream epr_file;
    ofstream epr_av_file;
    ofstream log_file;
    ios_base::openmode epr_av_t = ios::trunc;
    TeeDevice td(cout, log_file);
    TeeStream log(td);

    // Real space fields
    mat rho;
    mat Px;
    mat Py;

    // Set random seed
    arma_rng::set_seed_random();

    /* Parse program arguments */
    if (argc == 1) {
        // Too few input arguments
        cout << "Run with arguments: ./toner-tu-semi-implicit <in/out dir> -<options: c/C/a/v/n>" << endl;
        cout << "Options:" << endl;
        cout << "c: continue from files stored in in/out dir" << endl;
        cout << "C: continue from files, average EPR _AND_ continue averaging of EPR from where previous simulation stopped" << endl;
        cout << "   This flag automatically imposes flags a and c. Also requires the file epr_av.txt" << endl;
        cout << "   Provide step # of previous simulation with the flag" << endl;
        cout << "a: compute and average EPR" << endl;
        cout << "v: verbose mode, i.e. output full state (including EPR if this is computed) on every print interval" << endl;
        cout << "n: no noise - use this rather than setting D=0" << endl;
        cout << "x: use AXY parameters" << endl;
        cout << "E.g.: ./toner-tu-int-fact some_dir/ -c -a -v" << endl;
        cout << "E.g.: ./toner-tu-int-fact some_dir/ -C 1000000" << endl;
        return 0;
    }
    else {
        // First argument is input directory
        argc--;
        argv++;

        input_dir = *argv;
        if (!rdata(input_dir)) return 0;
        else {
            log_file.open(input_dir + "log.txt", ios::trunc);
            log << "Parameters read from in_data file:" << endl;
            log << "Steps: " << steps << endl;
            log << "Print interval: " << print_interval << endl << endl;
            log << "Nx: " << Nx << endl;
            log << "Ny: " << Ny << endl;
            log << "Lx: " << Lx << endl;
            log << "Ly: " << Ly << endl;
            log << "dt: " << dt << endl << endl;
            log << "D: " << Diff << endl;
            log << "lambda: " << l << endl;
            log << "kappa: " << k << endl;
            log << "w: " << w << endl;
            log << "w1: " << w1 << endl << endl;
            log << "rho0: " << rho0 << endl;
            log << "Px0: " << Px0 << endl;
            log << "Py0: " << Py0 << endl << endl;

            // If input is read successfully, initialize fields and epr
            rho.ones(Nx,Ny); rho *= rho0;
            Px.ones(Nx,Ny); Px *= Px0;
            Py.ones(Nx,Ny); Py *= Py0;

            sqrt_Ddt = sqrt(Diff*dt);
        }

        // Following arguments contains options
        while (argc > 1) {
            argc--;
            argv++;

            if (**argv == '-') {
                option = *(*argv + 1);

                if (option == 'c') {
                    log << "Reading initial state from file..." << endl << endl;
                    if (!(rho.load(input_dir + "rho.txt") && Px.load(input_dir + "Px.txt") && Py.load(input_dir + "Py.txt"))) {
                        log << "Could not read initial state from file, check files exist..." << endl;
                        return 0;
                    }
                }
                else if (option == 'a') {
                    average_epr = true;
                }
                else if (option == 'v') {
                    verbose = true;
                }
                else if (option == 'n') {
                    no_noise = true;
                }
                else if (option == 'x') {
                    log << "Running with parameters from AXY model" << endl << endl;
                    axy = true;
                    l = 3/16;
                    k = 5/16;
                    w = 1;
                    w1 = 1/2;

                    //mat zr(Nx,Ny,fill::randn);
                    //rho += Diff*zr;
                }
                else if (option == 'C') {
                    continue_average = true;
                    average_epr = true;
                    log << "Reading initial state from file..." << endl << endl;
                    if (!(rho.load(input_dir + "rho.txt") && Px.load(input_dir + "Px.txt") && Py.load(input_dir + "Py.txt"))) {
                        log << "Could not read initial state from file, check files exist..." << endl;
                        return 0;
                    }

                    if (argc > 1) {
                        argc--;
                        argv++;

                        average_shift = strtol(*argv, nullptr, 10);
                        if (average_shift == 0) {
                            log << "Invalid number of steps passed with flag -C" << endl;
                            return 0;
                        }
                        log << "Previous number of steps used in EPR average: " << average_shift << endl;
                    }
                    else {
                        log << "Flag -C requires specification of number of steps used in previous average" << endl;
                        return 0;
                    }
                    vec epr_av_vec;
                    if (!epr_av_vec.load(input_dir + "epr_av.txt")) {
                        log << "Could not load file epr_av.txt" << endl;
                        return 0;
                    }
                    epr_av = epr_av_vec[epr_av_vec.n_rows-1];
                    log << "Read the following values from epr_av.txt:" << endl;
                    log << "epr_av = " << epr_av << endl << endl;
                    epr_av_t = ios::app;
                }
                else {
                    log << "Unkown option: " << option << endl;
                    return 0;
                }
            }
            else {
                log << "Unknown program argument: " << *argv << endl;
                return 0;
            }
        }
    }

    /* Register signal handler function */
    signal(SIGINT, sigint_handler);

    // Convolution object
    Convolution conv;

    /* Initialize output files */
    // Polarization
    Px_file.open(input_dir + "Px_av.txt", ios::trunc);
    Py_file.open(input_dir + "Py_av.txt", ios::trunc);
    P_file.open(input_dir + "P_av.txt", ios::trunc);

    // Set precision of output files
    Px_file.precision(dbl::max_digits10);
    Py_file.precision(dbl::max_digits10);
    P_file.precision(dbl::max_digits10);

    // EPR
    if (average_epr) {
        epr_file.open(input_dir + "epr_ts.txt", epr_av_t);
        epr_av_file.open(input_dir + "epr_av.txt", epr_av_t);

        epr_file.precision(dbl::max_digits10);
        epr_av_file.precision(dbl::max_digits10);
    }

    /* Fields, initialization */
    // Fields Fourier space
    cx_mat rho_ft(Nx,Ny/2+1,fill::zeros);
    fft_forward(rho.memptr(), (MKL_Complex16 *)rho_ft.memptr());
    rho_ft.col(Ny/2).zeros();
    rho_ft.row(Nx/2).zeros();

    cx_mat Px_ft(Nx,Ny/2+1,fill::zeros);
    fft_forward(Px.memptr(), (MKL_Complex16 *)Px_ft.memptr());
    Px_ft.col(Ny/2).zeros();
    Px_ft.row(Nx/2).zeros();

    cx_mat Py_ft(Nx,Ny/2+1,fill::zeros);
    fft_forward(Py.memptr(), (MKL_Complex16 *)Py_ft.memptr());
    Py_ft.col(Ny/2).zeros();
    Py_ft.row(Nx/2).zeros();

    // EPR real space
    mat epr(Nx,Ny,fill::zeros);

    // EPR Fourier space
    cx_mat epr_ft(Nx,Ny/2+1,fill::zeros);

    // Intermediate stepping
    cx_mat rho_inter_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat Px_inter_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat Py_inter_ft(Nx,Ny/2+1,fill::zeros);

    /* Spatial derivatives Fourier space, initialization */
    // Polarization derivatives, first order
    cx_mat Px_x_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat Px_y_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat Py_x_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat Py_y_ft(Nx,Ny/2+1,fill::zeros);

    /* Nonlinearities Fourier space, initialization */
    // Quadratic and cubic polarization nonlinearities
    cx_mat Px_sq_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat Py_sq_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat P_sq_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat P_sq_Px_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat P_sq_Py_ft(Nx,Ny/2+1,fill::zeros);

    // Density-polarization nonlinearity
    cx_mat rho_Px_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat rho_Py_ft(Nx,Ny/2+1,fill::zeros);

    // Lambda and kappa nonlinearities
    cx_mat lambda_x_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat lambda_x1_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat lambda_x2_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat lambda_y_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat lambda_y1_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat lambda_y2_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat kappa_x_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat kappa_x1_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat kappa_x2_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat kappa_y_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat kappa_y1_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat kappa_y2_ft(Nx,Ny/2+1,fill::zeros);

    /* Noise fields, real and Fourier space, initialization */
    // Real space noise
    mat z1(Nx,Ny,fill::zeros);
    mat z2(Nx,Ny,fill::zeros);

    // Fourier space noise
    cx_mat z1_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat z2_ft(Nx,Ny/2+1,fill::zeros);

    /* Nonlinearities used to compute the EPR */
    cx_mat trs_odd_x_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat trs_odd_y_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat trs_even_x_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat trs_even_y_ft(Nx,Ny/2+1,fill::zeros);

    cx_mat epr_x1_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat epr_x2_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat epr_y1_ft(Nx,Ny/2+1,fill::zeros);
    cx_mat epr_y2_ft(Nx,Ny/2+1,fill::zeros);

    /* Wave vectors */
    vec qx_vec = join_cols(regspace<vec>(0,Nx/2),regspace<vec>(-Nx/2+1,-1));
    cx_mat qx(Nx,Ny/2+1,fill::ones);
    qx.each_col() %= complex<double>(0,1)*qx_vec/Lx;

    rowvec qy_vec = regspace<rowvec>(0,Ny/2);
    cx_mat qy(Nx,Ny/2+1,fill::ones);
    qy.each_row() %= complex<double>(0,1)*qy_vec/Ly;

    Mat<cx_double> q_sq = -(qx % qx) - (qy % qy);

    /* Mobility matrix M, N = (1+dt*M)^(-1) */
    cx_mat M_mat(3,3,fill::zeros);
    cx_mat N_mat(3,3,fill::zeros);

    cx_mat N_11(Nx,Ny/2+1,fill::zeros);
    cx_mat N_12(Nx,Ny/2+1,fill::zeros);
    cx_mat N_13(Nx,Ny/2+1,fill::zeros);
    cx_mat N_21(Nx,Ny/2+1,fill::zeros);
    cx_mat N_22(Nx,Ny/2+1,fill::zeros);
    cx_mat N_23(Nx,Ny/2+1,fill::zeros);
    cx_mat N_31(Nx,Ny/2+1,fill::zeros);
    cx_mat N_32(Nx,Ny/2+1,fill::zeros);
    cx_mat N_33(Nx,Ny/2+1,fill::zeros);

    for (int i=0; i<Nx; i++) {
        for (int j=0; j<Ny/2+1; j++) {
            M_mat(0,0) = 0;
            M_mat(0,1) = w*qx(i,j);
            M_mat(0,2) = w*qy(i,j);
            M_mat(1,0) = w1*qx(i,j);
            M_mat(1,1) = 1. + q_sq(i,j);
            M_mat(1,2) = 0;
            M_mat(2,0) = w1*qy(i,j);
            M_mat(2,1) = 0;
            M_mat(2,2) = 1. + q_sq(i,j);

            M_mat = eye(3,3) + dt*M_mat;
            N_mat = M_mat.i();

            N_11(i,j) = N_mat(0,0);
            N_12(i,j) = N_mat(0,1);
            N_13(i,j) = N_mat(0,2);
            N_21(i,j) = N_mat(1,0);
            N_22(i,j) = N_mat(1,1);
            N_23(i,j) = N_mat(1,2);
            N_31(i,j) = N_mat(2,0);
            N_32(i,j) = N_mat(2,1);
            N_33(i,j) = N_mat(2,2);
        }
    }

    if (verbose) {
        log << endl << "Printing initial state to file..." << endl;
        // Fourier transform
        fft_backward((MKL_Complex16 *) rho_ft.memptr(), rho.memptr());
        fft_backward((MKL_Complex16 *) Px_ft.memptr(), Px.memptr());
        fft_backward((MKL_Complex16 *) Py_ft.memptr(), Py.memptr());

        // Print output
        rho.save(input_dir + "rho_0.txt", raw_ascii);
        Px.save(input_dir + "Px_0.txt", raw_ascii);
        Py.save(input_dir + "Py_0.txt", raw_ascii);

        if (average_epr) {
            log << endl << "Printing spatiotemporally resolved EPR density to file..." << endl << endl;
            // Fourier transform
            fft_backward((MKL_Complex16 *) epr_ft.memptr(), epr.memptr());

            // Print output
            epr.save(input_dir + "epr_0.txt", raw_ascii);
        }
    }

    start = std::clock();
    for (int i=1; i<=steps; i++) {
        /* Compute derivatives */
        // Polarization
        Px_x_ft = qx%Px_ft;
        Px_y_ft = qy%Px_ft;
        Py_x_ft = qx%Py_ft;
        Py_y_ft = qy%Py_ft;

        /* Nonlinearities */
        // Quadratic and cubic polarization nonlinearities
        conv.convolve2d_same(Px_sq_ft,Px_ft);
        conv.convolve2d_same(Py_sq_ft,Py_ft);
        P_sq_ft = Px_sq_ft + Py_sq_ft;

        conv.convolve2d(P_sq_Px_ft,P_sq_ft,Px_ft);
        conv.convolve2d(P_sq_Py_ft,P_sq_ft,Py_ft);

        // Density-polarization nonlinearity
        conv.convolve2d(rho_Px_ft,rho_ft,Px_ft);
        conv.convolve2d(rho_Py_ft,rho_ft,Py_ft);

        // Lambda and kappa nonlinearities
        conv.convolve2d(lambda_x1_ft,Px_ft,Px_x_ft);
        conv.convolve2d(lambda_x2_ft,Py_ft,Px_y_ft);
        lambda_x_ft = lambda_x1_ft + lambda_x2_ft;

        conv.convolve2d(lambda_y1_ft,Px_ft,Py_x_ft);
        conv.convolve2d(lambda_y2_ft,Py_ft,Py_y_ft);
        lambda_y_ft = lambda_y1_ft + lambda_y2_ft;

        conv.convolve2d(kappa_x1_ft,Px_ft,Py_y_ft);
        conv.convolve2d(kappa_x2_ft,Py_ft,Py_x_ft);
        kappa_x_ft = kappa_x1_ft - kappa_x2_ft;

        conv.convolve2d(kappa_y1_ft,Py_ft,Px_x_ft);
        conv.convolve2d(kappa_y2_ft,Px_ft,Px_y_ft);
        kappa_y_ft = kappa_y1_ft - kappa_y2_ft;

        /* Calculate EPR first so that everything is done at same time step */
        if (average_epr) {
            // Calculate TRS odd and TRS even contributions to the EPR
            // TRS odd
            trs_odd_x_ft = qx%rho_ft; trs_odd_x_ft *= w1;
            trs_odd_x_ft += l*lambda_x_ft + k*kappa_x_ft;

            trs_odd_y_ft = qy%rho_ft; trs_odd_y_ft *= w1;
            trs_odd_y_ft += l*lambda_y_ft + k*kappa_y_ft;

            // TRS even
            trs_even_x_ft = Px_ft - rho_Px_ft + P_sq_Px_ft; trs_even_x_ft += q_sq%Px_ft;

            trs_even_y_ft = Py_ft - rho_Py_ft + P_sq_Py_ft; trs_even_y_ft += q_sq%Py_ft;

            // Calculate EPR
            epr_x1_ft = w*lambda_x_ft;
            conv.convolve2d(epr_x1_ft,epr_x1_ft,Px_ft);

            epr_y1_ft = w*lambda_y_ft;
            conv.convolve2d(epr_y1_ft,epr_y1_ft,Py_ft);

            conv.convolve2d(epr_x2_ft,trs_odd_x_ft,trs_even_x_ft);

            conv.convolve2d(epr_y2_ft,trs_odd_y_ft,trs_even_y_ft);

            epr_ft = - epr_x1_ft - epr_y1_ft - epr_x2_ft - epr_y2_ft;


            // Divide by noise coefficient D if we are solving with noise
            if (!no_noise) {
                epr_ft /= Diff;
            }

            // Average EPR 1
            epr_av = ((i+average_shift-1)*epr_av + epr_ft(0,0).real())/(i+average_shift);
        }

        /* Intermediate stepping */
        rho_inter_ft = rho_ft;

        if (axy) Px_inter_ft = rho_Px_ft/2 - P_sq_Px_ft/8 - l*lambda_x_ft - k*kappa_x_ft;
        else Px_inter_ft = rho_Px_ft - P_sq_Px_ft - l*lambda_x_ft - k*kappa_x_ft;
        Px_inter_ft *= dt;
        Px_inter_ft += Px_ft;

        if (axy) Py_inter_ft = rho_Py_ft/2 - P_sq_Py_ft/8 - l*lambda_y_ft - k*kappa_y_ft;
        else Py_inter_ft = rho_Py_ft - P_sq_Py_ft - l*lambda_y_ft - k*kappa_y_ft;
        Py_inter_ft *= dt;
        Py_inter_ft += Py_ft;

        // Sample noise
        if (!no_noise) {
            z1.randn(); fft_forward(z1.memptr(), (MKL_Complex16 *) z1_ft.memptr());
            z2.randn(); fft_forward(z2.memptr(), (MKL_Complex16 *) z2_ft.memptr());

            z1.col(Ny/2).zeros(); z1.row(Nx/2).zeros();
            z2.col(Ny/2).zeros(); z2.row(Nx/2).zeros();

            Px_inter_ft += sqrt_Ddt * z1_ft;
            Py_inter_ft += sqrt_Ddt * z2_ft;
        }

        /* Final stepping */
        // Step
        rho_ft = N_11%rho_inter_ft; rho_ft += N_12%Px_inter_ft; rho_ft += N_13%Py_inter_ft;

        Px_ft = N_21%rho_inter_ft; Px_ft += N_22%Px_inter_ft; Px_ft += N_23%Py_inter_ft;

        Py_ft = N_31%rho_inter_ft; Py_ft += N_32%Px_inter_ft; Py_ft += N_33%Py_inter_ft;

        /* Print output */
        if (i % print_interval == 0) {
            duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
            log << "============ UPDATE ============" << endl;
            log << "Simulation directory: " << input_dir << endl;
            log << "Step: " << i << endl;
            if (continue_average) log << "(Step shifted by: " << average_shift << ")" << endl;
            log << "Time spent since last update: " << duration << endl;

            log << "Px: " << Px_ft(0,0).real() << endl;
            Px_file << Px_ft(0,0).real() << endl;

            log << "Py: " << Py_ft(0,0).real() << endl;
            Py_file << Py_ft(0,0).real() << endl;

            log << "|<P>|: " << sqrt(Px_ft(0,0).real()*Px_ft(0,0).real() + Py_ft(0,0).real()*Py_ft(0,0).real()) << endl;
            log << "sqrt(<P^2>): " << sqrt(P_sq_ft(0,0).real()) << endl;
            P_file << sqrt(P_sq_ft(0,0).real()) << endl;

            if (verbose) {
                log << endl << "Printing state to file..." << endl;
                // Fourier transform
                fft_backward((MKL_Complex16 *) rho_ft.memptr(), rho.memptr());
                fft_backward((MKL_Complex16 *) Px_ft.memptr(), Px.memptr());
                fft_backward((MKL_Complex16 *) Py_ft.memptr(), Py.memptr());

                // Print output
                rho.save(input_dir + "rho_" + to_string(i) + ".txt", raw_ascii);
                Px.save(input_dir + "Px_" + to_string(i) + ".txt", raw_ascii);
                Py.save(input_dir + "Py_" + to_string(i) + ".txt", raw_ascii);
            }
            if (average_epr) {
                log << endl << "===== EPR =====" << endl;
                log << "Average EPR: " << epr_av << endl;
                epr_av_file << epr_av << endl;

                log << "Instantaneous EPR (time series): " << epr_ft(0,0).real() << endl;
                epr_file << epr_ft(0,0).real() << endl;

                if (verbose) {
                    log << endl << "Printing spatiotemporally resolved EPR density to file..." << endl;
                    // Fourier transform
                    fft_backward((MKL_Complex16 *) epr_ft.memptr(), epr.memptr());

                    // Print output
                    epr.save(input_dir + "epr_" + to_string(i) + ".txt", raw_ascii);
                }
            }
            log << "================================" << endl << endl;
            start = std::clock();
        }
        if (sigint_flag) {
            log << endl << "Received SIGINT, exiting gracefully..." << endl;
            break;
        }
    }

    /* Output final state */
    // Fourier transform final config
    fft_backward((MKL_Complex16 *) rho_ft.memptr(), rho.memptr());
    fft_backward((MKL_Complex16 *) Px_ft.memptr(), Px.memptr());
    fft_backward((MKL_Complex16 *) Py_ft.memptr(), Py.memptr());

    // Save final state to file
    log << "Printing final state to file ..." << endl;
    rho.save(input_dir + "rho.txt", raw_ascii);
    Px.save(input_dir + "Px.txt", raw_ascii);
    Py.save(input_dir + "Py.txt", raw_ascii);

    // Save final EPR state to file
    if (average_epr) {
        fft_backward((MKL_Complex16 *) epr_ft.memptr(), epr.memptr());
        epr.save(input_dir + "epr.txt", raw_ascii);
    }

    /* Close output files */
    // Polarization
    Px_file.close();
    Py_file.close();
    P_file.close();

    // Log stream
    log.close();
    log_file.close();

    // EPR
    if (average_epr) {
        epr_file.close();
        epr_av_file.close();
    }

    conv.cleanup();

    return 0;
}

void sample_noise(cx_double *z) {
    double a, b;

    for (int j=0; j<Ny/2; j++) {
        for (int i=0; i<Nx/2; i++) {
            if (j == 0) {
                if (i == 0) {
                    z[0] = randn<double>();
                }
                else {
                    a = randn<double>(); b = randn<double>();
                    z[i] = a + complex<double>(0,1)*b;
                    z[Nx-i] = a - complex<double>(0,1)*b;
                }
            }
            else {
                z[i + Nx*j] = randn<double>() + complex<double>(0,1)*randn<double>();
            }
        }
        for (int i=Nx/2+1; i<Nx; i++) {
            if (j == 0) {
                continue;
            }
            else {
                z[i + Nx*j] = randn<double>() + complex<double>(0,1)*randn<double>();
            }
        }
    }
}

Convolution::Convolution() {
    /* Create FFT descriptor handles */
    create_descriptor_handles(Nx,Ny);

    /* Field initialization */
    // Fields used for convolutions and paddings (global)
    conv_out_1.zeros(3*Nx/2,3*Ny/2);
    conv_out_2.zeros(3*Nx/2,3*Ny/2);
    conv_out_ft.zeros(3*Nx/2,3*Ny/4+1);
    pad_ft1.zeros(3*Nx/2,3*Ny/4+1);
    pad_ft2.zeros(3*Nx/2,3*Ny/4+1);
}

void Convolution::convolve2d(Mat<cx_double> &out, Mat<cx_double> &A, Mat<cx_double> &B) {
    // Pad matrices A and B
    pad_matrix(pad_ft1.memptr(),A.memptr());
    pad_matrix(pad_ft2.memptr(),B.memptr());

    // Transform to real space
    fft_padded_backward((MKL_Complex16 *) pad_ft1.memptr(), conv_out_1.memptr());
    fft_padded_backward((MKL_Complex16 *) pad_ft2.memptr(), conv_out_2.memptr());

    // Compute real space product
    conv_out_1 %= conv_out_2;

    // Transform back == convolution in Fourier space
    fft_padded_forward(conv_out_1.memptr(), (MKL_Complex16 *) conv_out_ft.memptr());

    // Shed off padding and zero out final row/column
    remove_pad(out.memptr(),conv_out_ft.memptr());
    out.col(Ny/2).zeros();
    out.row(Nx/2).zeros();
}

void Convolution::convolve2d_same(Mat<cx_double> &out, Mat<cx_double> &A) {
    // Pad matrix A
    pad_matrix(pad_ft1.memptr(),A.memptr());

    // Transform to real space
    fft_padded_backward((MKL_Complex16 *) pad_ft1.memptr(), conv_out_1.memptr());

    // Compute real space product
    conv_out_1 %= conv_out_1;

    // Transform back == convolution in Fourier space
    fft_padded_forward(conv_out_1.memptr(), (MKL_Complex16 *) conv_out_ft.memptr());

    // Shed off padding and zero out final row/column
    remove_pad(out.memptr(),conv_out_ft.memptr());

    out.col(Ny/2).zeros();
    out.row(Nx/2).zeros();
}

void Convolution::pad_matrix(cx_double *out, cx_double *in) {
    for (int i=0; i<Nx/2; i++) {
        for (int j=0; j<Ny/2; j++) {
            out[i + 3*Nx*j/2] = in[i + Nx*j];
        }
    }
    for (int i=Nx+1; i<3*Nx/2; i++) {
        for (int j=0; j<Ny/2; j++) {
            out[i + 3*Nx*j/2] = in[i-Nx/2 + Nx*j];
        }
    }
}

void Convolution::remove_pad(cx_double *out, cx_double *in) {
    for (int i=0; i<Nx/2; i++) {
        for (int j=0; j<Ny/2; j++) {
            out[i + Nx*j] = in[i + 3*Nx*j/2];
        }
    }
    for (int i=Nx+1; i<3*Nx/2; i++) {
        for (int j=0; j<Ny/2; j++) {
            out[i-Nx/2 + Nx*j] = in[i + 3*Nx*j/2];
        }
    }
}

void Convolution::cleanup() {
    // Free MKL FFT descriptor handles (calls c subroutine)
    free_descriptor_handles();
}

bool rdata(const string &input_dir) {
    ifstream input(input_dir + "in_data");
    string delimiter = "=";
    string line;
    string var;
    bool steps_b = false;
    bool print_interval_b = false;
    bool Nx_b = false;
    bool Ny_b = false;
    bool Lx_b = false;
    bool Ly_b = false;
    bool dt_b = false;
    bool Diff_b = false;
    bool l_b = false;
    bool w_b = false;
    bool rho0_b = false;
    bool Px0_b = false;
    bool Py0_b = false;

    if(input.is_open()) {
        while (!input.eof()) {
            getline(input,line);
            if (line.find(delimiter) == string::npos) continue;
            else {
                var = line.substr(0, line.find(delimiter));
                var.erase(remove_if(var.begin(), var.end(), ::isspace), var.end());

                line.erase(0, line.find(delimiter) + delimiter.length());
                line.erase(remove_if(line.begin(), line.end(), ::isspace), line.end());

                if (var == "steps") {
                    steps = atoi(line.c_str());
                    steps_b = true;
                }
                else if (var == "pinterval") {
                    print_interval = atoi(line.c_str());
                    print_interval_b = true;
                }
                else if (var == "Nx") {
                    Nx = atoi(line.c_str());
                    Nx_b = true;
                }
                else if (var == "Ny") {
                    Ny = atoi(line.c_str());
                    Ny_b = true;
                }
                else if (var == "Lx") {
                    Lx = stod(line);
                    Lx_b = true;
                }
                else if (var == "Ly") {
                    Ly = stod(line);
                    Ly_b = true;
                }
                else if (var == "dt") {
                    dt = stod(line);
                    dt_b = true;
                }
                else if (var == "D") {
                    Diff = stod(line);
                    Diff_b = true;
                }
                else if (var == "lambda") {
                    l = stod(line);
                    k = l;
                    l_b = true;
                }
                else if (var == "w") {
                    w = stod(line);
                    w1 = w/2;
                    w_b = true;
                }
                else if (var == "rho0") {
                    rho0 = stod(line);
                    rho0_b = true;
                }
                else if (var == "Px0") {
                    Px0 = stod(line);
                    Px0_b = true;
                }
                else if (var == "Py0") {
                    Py0 = stod(line);
                    Py0_b = true;
                }
                else {
                    cout << "Unknown input parameter: " << var << endl;
                    return false;
                }
            }
        }
        if (!(steps_b && print_interval_b && Nx_b && Ny_b && Lx_b && Ly_b && dt_b && Diff_b && l_b && w_b && rho0_b && Px0_b && Py0_b)) {
            cout << "Could not find all parameters..." << endl;
            return false;
        }
    }
    else return false;
    return true;
}
