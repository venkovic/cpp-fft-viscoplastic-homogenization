#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <math.h>
#include <vector>
#include <string>
#include <algorithm>
#include <fftw3.h>
#include <stdio.h>
using namespace std;

//
// From cmd: 
// g++ -I .. -o exec_name filename.cpp -Wall -std=c++11 -lfftw3 -lm
//
// From Geany:
// Compile:	g++ -c "%f" -Wall -std=c++11 -lfftw3 -lm
//   Build: g++ -o "%e" "%f" -Wall -std=c++11 -lfftw3 -lm
//
//		-I path						   : include a directory where header files are to be found
//		-Wall						   : all warnings flag
//		-o 						   : optimization flag
// (Geany)	-o "%e" "%f"					   : optimization flag -- Make the execution file with same name as the cpp source file
// (cmd)	-o exec_name filename.cpp "%f" 			   : optimization flag -- output directed to file %f
// (Geany)	-c "%f"						   : compilation flag // without -c, everything necessary to compile AND link should be provided
//		-std=c++11					   : specifies cpp version
//		-fopenmp					   : uses openmp library
//
//	Linkers:
//			-lm		: math library
//			-lfftw3_omp	: fftw3_omp library
//			-lfftw3		: fftw3 library
//
//

// DATA STRUCTURES:
// 4-th order major-minor symmetric tensor
struct T4 {
	double r1111, r1122, r1112;
	double r2222, r2212;
	double r1212;
};
// 4-th order major-minor symmetric tensor field
struct T4_field {
	vector<double> r1111, r1122, r1112;
	vector<double> r2222, r2212;
	vector<double> r1212;
};
// Numerical solution
struct sol {
	int n;
	double * sig_11, * sig_22, * sig_12;
	double * eps_11, * eps_22, * eps_12;
	vector<double> err;
	double etol;
	int niter;
};

// UTILITIES:
//
// split a string
vector<string> split(const string &s, char delim) {
    stringstream ss(s);
    string item;
    vector<string> tokens;
    while (getline(ss, item, delim)) {
        tokens.push_back(item);
    }
    return tokens;
}

vector<double> inv_mat_3by3(vector<double> a) {
	double a11=a[0], a12=a[1], a13=a[2];
	double a21=a[3], a22=a[4], a23=a[5];
	double a31=a[6], a32=a[7], a33=a[8];
	double det=a11*a22*a33+a21*a32*a13+a31*a12*a23-a11*a32*a23-a31*a22*a13-a21*a12*a33;
	double b11=(a22*a33-a23*a32)/det, b12=(a13*a32-a12*a33)/det, b13=(a12*a23-a13*a22)/det;
	double b21=(a23*a31-a21*a33)/det, b22=(a11*a33-a13*a31)/det, b23=(a13*a21-a11*a23)/det;
	double b31=(a21*a32-a22*a31)/det, b32=(a12*a31-a11*a32)/det, b33=(a11*a22-a12*a21)/det;
	return {b11,b12,b13,b21,b22,b23,b31,b32,b33};
}

int sgn(double x) {
	if (x > 0) return 1;
	if (x < 0) return -1;
	return 0;
}

// SUBROUTINES:
// Compute components of the discete Green operator in Fourier space
T4_field set_discrete_green_operator(int n, T4 L0) {
	//  n: Number of frequencies in each direction -- must be even.
	// L0: Reference stiffness.
	//
	// Wave vectors 
	vector<double> q, t, c, s;
	q.push_back(0.);
	t.push_back(tan(q.back()/2.));
	c.push_back(cos(q.back()));
	s.push_back(sin(q.back()));
	for (int i=1;i<n/2+1;i++) {
		q.push_back(2.*M_PI/n*i);
		t.push_back(tan(q.back()/2.));
		c.push_back(cos(q.back()));
		s.push_back(sin(q.back()));
	}
	for (int i=-n/2+1;i<0;i++) {
		q.push_back(2.*M_PI/n*i);
		t.push_back(tan(q.back()/2.));
		c.push_back(cos(q.back()));
		s.push_back(sin(q.back()));	
	}
	//
	// Discrete acoustic tensor K, and its inverse N
	int n2=n/2+1;
	double K11, K12, K22, det_K;
	vector<double> N11(n*n2), N12(n*n2), N22(n*n2);
	for (int i=0;i<n;i++) {
		for (int j=0;j<n2;j++) {
			int k=n2*i+j;
			if ((i==0)&(j==0)) {
				N11[k]=0.; N22[k]=0.; N12[k]=0.;
			}
			else if ((i==n/2)&(j==n/2)) {
				N11[k]=0.; N22[k]=0.; N12[k]=0.;
			}
			else {
				K11=2.*s[i]*s[j]*L0.r1112+(1.-c[i])*(1.+c[j])*L0.r1111+(1.+c[i])*(1.-c[j])*L0.r1212;
				K22=2.*s[i]*s[j]*L0.r2212+(1.-c[i])*(1.+c[j])*L0.r1212+(1.+c[i])*(1.-c[j])*L0.r2222;
				K12=s[i]*s[j]*(L0.r1122+L0.r1212)+(1.-c[i])*(1.+c[j])*L0.r1112+(1.+c[i])*(1.-c[j])*L0.r2212;
				det_K=K11*K22-K12*K12;
				N11[k]=K22/det_K; N22[k]=K11/det_K;
				N12[k]=-K12/det_K;
			}
		}
	}
	//
	// Discrete Green operator
	vector<double> Gp1111(n*n2), Gp1122(n*n2), Gp1112(n*n2);
	vector<double> Gp2222(n*n2), Gp2212(n*n2);
	vector<double> Gp1212(n*n2);
	for (int i=0;i<n;i++) {
		for (int j=0;j<n2;j++) {
			int k = n2*i+j;
			Gp1111[k]=(1.-c[i])*(1+c[j])*N11[k];
			Gp1122[k]=s[i]*s[j]*N12[k];
			Gp1112[k]=.5*(s[i]*s[j]*N11[k]+(1.-c[i])*(1.+c[j])*N12[k]);
			Gp2222[k]=(1.+c[i])*(1.-c[j])*N22[k];
			Gp2212[k]=.5*(s[i]*s[j]*N22[k]+(1.+c[i])*(1.-c[j])*N12[k]);
			Gp1212[k]=.25*((1.+c[i])*(1.-c[j])*N11[k]+2.*s[i]*s[j]*N12[k]+(1.-c[i])*(1.+c[j])*N22[k]);
		}
	}
	T4_field hGp;
	hGp.r1111=Gp1111; hGp.r1122=Gp1122; hGp.r1112=Gp1112;
	hGp.r2222=Gp2222; hGp.r2212=Gp2212;
	hGp.r1212=Gp1212;
	return hGp;
}



//
// Material instance
struct material {
	int type;
	double sig11_k, sig22_k, sig12_k;
	double eps11_k, eps22_k, eps12_k;
	double dp=0;
	//
	// type=0: 2D elastic medium
	T4 L;				// elastic stiffness
	//
	// type=1: 2D von Mises perfect elastoplastic medium
	double sig0;		// yield stress
	double mu2d;		
	int plastified=0;	
	//
	// type=2: 2D von Mises elastoplastic medium with linear isotropic hardening
	double h, sig0_t;
	//
	// type=3: 2D von Mises perfect elastoviscoplastic medium with power law strain-rate sensitivity
	double p=0;			// equivalent plastic strain
	double k;			// viscosity 
	double m;			// strain-rate sensitivity
	//
	// type=4: 2D von Mises perfect elastoviscoplastic medium with power law strain-rate sensitivity
	double tau;			// relaxation time 
	double e;			// strain-rate sensitivity
	//
	// type=5 2D isotropic crystal with two slip systems
	int nslip=2;
	vector<double> t0;		// critical shear strenght
	vector<double> th;		// angles of slip systems
	//double m;				// strain-rate sensitivity
	//double h;				// hardening coefficient
	vector<double> dgam0;	// slip rate
	//
	vector<double> m11, m22, m12;
	vector<double> C11rs_mrs, C22rs_mrs, C12rs_mrs;
	vector<double> C11rs_mrs_m11, C11rs_mrs_m22, C11rs_mrs_m12;
	vector<double> C22rs_mrs_m11, C22rs_mrs_m22, C22rs_mrs_m12;
	vector<double> C12rs_mrs_m11, C12rs_mrs_m22, C12rs_mrs_m12;
	vector<double> gam;		// plastic slip
	vector<double> Dgam;	// plastic slip increment
};
//
// Initialize material instance
void ini_mat(material &mat, string kline) {
	mat.sig11_k=0.; mat.sig22_k=0.; mat.sig12_k=0.;
	mat.eps11_k=0.; mat.eps22_k=0.; mat.eps12_k=0.;
	//
	vector<string> x=split(kline,','); // split line between ','
	mat.type=atoi(x[0].c_str());
	//
	mat.L.r1111=atof(x[1].c_str()); mat.L.r1122=atof(x[2].c_str()); mat.L.r1112=atof(x[3].c_str());
	mat.L.r2222=atof(x[4].c_str()); mat.L.r2212=atof(x[5].c_str());
	mat.L.r1212=atof(x[6].c_str());	
	//
	// 2D von Mises perfect elastoplastic medium
	if (mat.type==1) {
		/*
		mat.sig0=atof(x[7].c_str());
		mat.mu2d=mat.L.r1212;
		*/
		mat.sig0=atof(x[7].c_str());
	}
	// 2D von Mises perfect elastoplastic medium
	else if (mat.type==2) {
		mat.sig0=atof(x[7].c_str());
		mat.h=atof(x[8].c_str());
		mat.sig0_t=mat.sig0;
		mat.mu2d=mat.L.r1212;
	}
	// 2D von Mises perfect viscoplastic
	else if (mat.type==4) {
		mat.sig0=atof(x[7].c_str());
		mat.tau=atof(x[8].c_str());
		mat.e=atof(x[9].c_str());
		mat.mu2d=mat.L.r1212;
	}
	// 2D isotropic crystal with two slip systems
	else if (mat.type==5) {
		mat.t0=vector<double>(mat.nslip);
		mat.th=vector<double>(mat.nslip);
		mat.dgam0=vector<double>(mat.nslip);
		mat.m11=vector<double>(mat.nslip);		
		mat.m22=vector<double>(mat.nslip);		
		mat.m12=vector<double>(mat.nslip);		
		mat.C11rs_mrs=vector<double>(mat.nslip);	
		mat.C22rs_mrs=vector<double>(mat.nslip);	
		mat.C12rs_mrs=vector<double>(mat.nslip);		
		mat.C11rs_mrs_m11=vector<double>(mat.nslip);		
		mat.C11rs_mrs_m22=vector<double>(mat.nslip);	
		mat.C11rs_mrs_m12=vector<double>(mat.nslip);	
		mat.C22rs_mrs_m11=vector<double>(mat.nslip);		
		mat.C22rs_mrs_m22=vector<double>(mat.nslip);	
		mat.C22rs_mrs_m12=vector<double>(mat.nslip);	
		mat.C12rs_mrs_m11=vector<double>(mat.nslip);		
		mat.C12rs_mrs_m22=vector<double>(mat.nslip);	
		mat.C12rs_mrs_m12=vector<double>(mat.nslip);
		mat.Dgam=vector<double>(mat.nslip);
		//		
		mat.t0[0]=atof(x[7].c_str());
		mat.t0[1]=mat.t0[0];
		mat.th[0]=atof(x[8].c_str());
		mat.th[1]=mat.th[0]+M_PI/3.;
		mat.m=atof(x[9].c_str());
		mat.h=atof(x[10].c_str());
		mat.dgam0[0]=atof(x[11].c_str());
		mat.dgam0[1]=mat.dgam0[0];
		//
		for (int al=0;al<mat.nslip;al++) {
			mat.m11[al]=-cos(mat.th[al])*sin(mat.th[al]);
			mat.m22[al]= cos(mat.th[al])*sin(mat.th[al]);
			mat.m12[al]=.5*(pow(cos(mat.th[al]),2)-pow(sin(mat.th[al]),2));
			//
			mat.C11rs_mrs[al]=mat.L.r1111*mat.m11[al]+mat.L.r1122*mat.m22[al]+2.*mat.L.r1112*mat.m12[al];
			mat.C22rs_mrs[al]=mat.L.r1122*mat.m11[al]+mat.L.r2222*mat.m22[al]+2.*mat.L.r2212*mat.m12[al];
			mat.C12rs_mrs[al]=mat.L.r1112*mat.m11[al]+mat.L.r2212*mat.m22[al]+2.*mat.L.r1212*mat.m12[al];
			//
			mat.C11rs_mrs_m11[al]=mat.C11rs_mrs[al]*mat.m11[al];
			mat.C11rs_mrs_m22[al]=mat.C11rs_mrs[al]*mat.m22[al];
			mat.C11rs_mrs_m12[al]=mat.C11rs_mrs[al]*mat.m12[al];
			mat.C22rs_mrs_m11[al]=mat.C22rs_mrs[al]*mat.m11[al];
			mat.C22rs_mrs_m22[al]=mat.C22rs_mrs[al]*mat.m22[al];
			mat.C22rs_mrs_m12[al]=mat.C22rs_mrs[al]*mat.m12[al];
			mat.C12rs_mrs_m11[al]=mat.C12rs_mrs[al]*mat.m11[al];
			mat.C12rs_mrs_m22[al]=mat.C12rs_mrs[al]*mat.m22[al];
			mat.C12rs_mrs_m12[al]=mat.C12rs_mrs[al]*mat.m12[al];		
		}	
	}
}




//
// Write output file for 2nd order tensor field
void write_output(double* a11,double* a22,double* a12,int n,string fname) {
	ofstream myfile;
	myfile.open(fname);
	myfile << n << "," << n << endl;
	for (int i=0;i<n*n;++i) {
		myfile << a11[i] << "," << a22[i] << "," << a12[i] << endl;
	}
	myfile.close();
}
//
// Write output file for voids and plastified regions 
void write_void_and_plastified_output(int n,vector<material> mat, string fname) {
	ofstream myfile;
	myfile.open(fname);
	myfile << n << "," << n << endl;
	for (int k=0;k<n*n;++k) {
		if ((mat[k].type!=0)&(mat[k].plastified==false)) {
			myfile << "0" << endl;			
		}
		else {
			myfile << "1" << endl;
		}
	}
	myfile.close();
}
//
// Write output file equivalent plastic strain
void write_p_output(int n,vector<material> mat, string fname) {
	ofstream myfile;
	myfile.open(fname);
	myfile << n << "," << n << endl;
	for (int k=0;k<n*n;++k) {
		myfile << mat[k].p << endl;
	}
	myfile.close();
}
//
// Write output file for 4th order real tensor field
void write_green(T4_field T, int n, string fname) {
	ofstream myfile;
	myfile.open(fname);
	myfile << n << "," << n << endl;
	for (int i=0;i<n*(n/2+1);++i) {
		myfile << T.r1111[i] << "," << T.r1122[i] << "," << T.r1112[i] << "," << T.r2222[i] << "," << T.r2212[i] << "," << T.r1212[i] << "\n";
	}
	myfile.close();
}
//
// Write error output file
void write_output(vector<double> err,int niter,string fname) {
	ofstream myfile;
	myfile.open(fname);
	myfile << niter << endl;
	for (int i=0;i<niter+1;++i) {
		myfile << err[i] << endl;
	}
	myfile.close();
}
//
// Read geo file
vector<int> read_geo(string fname) {
	ifstream myfile;
	myfile.open(fname);
	string line;
	getline(myfile,line);
	vector<string> x=split(line,',');
	int nx=atoi(x[0].c_str());
	int ny=atoi(x[1].c_str());
	vector<int> geo(nx*ny);
	for (int k=0;k<nx*ny;k++) {
		getline(myfile,line);
		geo[k]=atoi(line.c_str());
	}
	myfile.close();
	return geo;
}
//
// Read mat file
vector<T4> read_mat(string fname) {
	ifstream myfile;
	myfile.open(fname);
    string line;
    getline(myfile,line);
    int nmat=atoi(line.c_str());
	vector<T4> L(nmat);
    for (int imat=0;imat<nmat;imat++) {
		getline(myfile,line);
		vector<string> x=split(line,',');
		L[imat].r1111=atof(x[0].c_str());
		L[imat].r1122=atof(x[1].c_str());
		L[imat].r1112=atof(x[2].c_str());
		L[imat].r2222=atof(x[3].c_str());
		L[imat].r2212=atof(x[4].c_str());
		L[imat].r1212=atof(x[5].c_str());
	}
	myfile.close();
	return L;
}
//
// Read eps_av file
vector<vector<double>> read_eps_av(string fname) {
	ifstream myfile;
	myfile.open(fname);
    string line;
    getline(myfile,line);
    int nt=atoi(line.c_str());
	vector<double> eps11(nt+1), eps22(nt+1), eps12(nt+1), time(nt+1);
    for (int tk=0;tk<nt+1;tk++) {
		getline(myfile,line);
		vector<string> x=split(line,',');
		eps11[tk]=atof(x[0].c_str());
		eps22[tk]=atof(x[1].c_str());
		eps12[tk]=atof(x[2].c_str());
		time[tk]=atof(x[3].c_str());
	}
	myfile.close();
	return {eps11, eps22, eps12, time};
}


//
// Read green file
T4_field read_green(string fname) {
	ifstream myfile;
	myfile.open(fname);
	string line;	
	getline(myfile,line);
	vector<string> x=split(line,',');
	int n=atoi(x[0].c_str());
	int ny=atoi(x[1].c_str());
	int n2=ny/2+1;
	vector<double> G1111(n*n2), G1122(n*n2), G1112(n*n2);
	vector<double> G2222(n*n2), G2212(n*n2);
	vector<double> G1212(n*n2);
	for (int k=0;k<n*n2;k++) {
		getline(myfile,line);
		vector<string> x=split(line,',');
		G1111[k]=atof(x[0].c_str());
		G1122[k]=atof(x[1].c_str());
		G1112[k]=atof(x[2].c_str());
		G2222[k]=atof(x[3].c_str());
		G2212[k]=atof(x[4].c_str());
		G1212[k]=atof(x[5].c_str());
	}
	myfile.close();
	T4_field hG;
	hG.r1111=G1111; hG.r1122=G1122; hG.r1112=G1112;
	hG.r2222=G2222; hG.r2212=G2212;
	hG.r1212=G1212;
	return hG;
}
//
// Direct solver
sol direct_solver(int n, T4_field hG, vector<int> I, vector<T4> L, double eps_av_11, double eps_av_22, double eps_av_12, double etol) {
	// 		   n: Number of frequencies in each direction -- must be even.
	// 		   L: Stiffness tensor field.
	// eps_av_11: Average axial strain prescribed along e_1.
	// eps_av_22: Average axial strain prescribed along e_2.
	// eps_av_12: Average shear strain prescribed along e_1 and e_2.
	// 		etol: Tolerance for the error in equlibrium.
	//
	int n2 = n/2+1;
	int it = 0;
	double cte = 1./(n*n);
	//vector<double> m0_sig_11(n*n), m0_sig_22(n*n), m0_sig_12(n*n);
	double* m0_sig_11 = static_cast<double*>(fftw_malloc(n*n*sizeof(double)));
	double* m0_sig_22 = static_cast<double*>(fftw_malloc(n*n*sizeof(double)));
	double* m0_sig_12 = static_cast<double*>(fftw_malloc(n*n*sizeof(double)));
	double* m_sig_11 = static_cast<double*>(fftw_malloc(n*n*sizeof(double)));
	double* m_sig_22 = static_cast<double*>(fftw_malloc(n*n*sizeof(double)));
	double* m_sig_12 = static_cast<double*>(fftw_malloc(n*n*sizeof(double)));
	double* m_deps_11 = static_cast<double*>(fftw_malloc(n*n*sizeof(double)));
	double* m_deps_22 = static_cast<double*>(fftw_malloc(n*n*sizeof(double)));
	double* m_deps_12 = static_cast<double*>(fftw_malloc(n*n*sizeof(double)));
	//vector<vector<double>> m_hdeps_11(n*n2,vector<double>(2)), m_hdeps_22(n*n2,vector<double>(2)), m_hdeps_12(n*n2,vector<double>(2));

	fftw_complex* m_hdeps_11 = static_cast<fftw_complex*>(fftw_malloc(n*n2*sizeof(fftw_complex)));
	fftw_complex* m_hdeps_22 = static_cast<fftw_complex*>(fftw_malloc(n*n2*sizeof(fftw_complex)));
	fftw_complex* m_hdeps_12 = static_cast<fftw_complex*>(fftw_malloc(n*n2*sizeof(fftw_complex)));

	fftw_complex* m_hsig_11 = static_cast<fftw_complex*>(fftw_malloc(n*n2*sizeof(fftw_complex)));
	fftw_complex* m_hsig_22 = static_cast<fftw_complex*>(fftw_malloc(n*n2*sizeof(fftw_complex)));
	fftw_complex* m_hsig_12 = static_cast<fftw_complex*>(fftw_malloc(n*n2*sizeof(fftw_complex)));
	fftw_complex* m_hdeps_11_tmp = static_cast<fftw_complex*>(fftw_malloc(n*n2*sizeof(fftw_complex)));
	fftw_complex* m_hdeps_22_tmp = static_cast<fftw_complex*>(fftw_malloc(n*n2*sizeof(fftw_complex)));
	fftw_complex* m_hdeps_12_tmp = static_cast<fftw_complex*>(fftw_malloc(n*n2*sizeof(fftw_complex)));
	//
	// Wave vectors
	double q;
	vector<double> t, c;
	q = 0.;
	t.push_back(tan(q/2.));
	c.push_back(cos(q));
	for (int i=1;i<n/2+1;i++) {
		q =2.*M_PI/n*i;
		t.push_back(tan(q/2.));
		c.push_back(cos(q));
	}
	for (int i=-n/2+1;i<0;i++) {
		q= 2.*M_PI/n*i;
		t.push_back(tan(q/2.));
		c.push_back(cos(q));
	}
	//
	// Set up FFTW plans
	int flags1 = FFTW_ESTIMATE;
	int flags2 = FFTW_ESTIMATE | FFTW_BACKWARD;
	fftw_plan fft_sig_11 = fftw_plan_dft_r2c_2d(n,n,m_sig_11,m_hsig_11,flags1);
	fftw_plan fft_sig_22 = fftw_plan_dft_r2c_2d(n,n,m_sig_22,m_hsig_22,flags1);
	fftw_plan fft_sig_12 = fftw_plan_dft_r2c_2d(n,n,m_sig_12,m_hsig_12,flags1);
	fftw_plan ifft_hdeps_11 = fftw_plan_dft_c2r_2d(n,n,m_hdeps_11_tmp,m_deps_11,flags2);
	fftw_plan ifft_hdeps_22 = fftw_plan_dft_c2r_2d(n,n,m_hdeps_22_tmp,m_deps_22,flags2);
	fftw_plan ifft_hdeps_12 = fftw_plan_dft_c2r_2d(n,n,m_hdeps_12_tmp,m_deps_12,flags2);
	//
	// Initiliaze 
	//
	for (int k=0;k<n*n2;k++) {
		m_hdeps_11[k][0]=0.; m_hdeps_11[k][1]=0.;
		m_hdeps_22[k][0]=0.; m_hdeps_22[k][1]=0.;
		m_hdeps_12[k][0]=0.; m_hdeps_12[k][1]=0.;
	}
	//
	for (int k=0;k<n*n;k++) {
		m0_sig_11[k]=L[I[k]].r1111*eps_av_11+L[I[k]].r1122*eps_av_22+2.*L[I[k]].r1112*eps_av_12;
		m0_sig_22[k]=L[I[k]].r1122*eps_av_11+L[I[k]].r2222*eps_av_22+2.*L[I[k]].r2212*eps_av_12;
		m0_sig_12[k]=L[I[k]].r1112*eps_av_11+L[I[k]].r2212*eps_av_22+2.*L[I[k]].r1212*eps_av_12;
		m_sig_11[k]=m0_sig_11[k]; m_sig_22[k]=m0_sig_22[k]; m_sig_12[k]=m0_sig_12[k];
	}
	//
	fftw_execute(fft_sig_11); fftw_execute(fft_sig_22); fftw_execute(fft_sig_12);
	//	
	vector<double> e;
	e.push_back((pow(t[0]*m_hsig_11[0][0]+t[0]*m_hsig_12[0][0],2)+pow(t[0]*m_hsig_12[0][0]+t[0]*m_hsig_22[0][0],2))*(1.+c[0])*(1.+c[0]));
	for (int i=0;i<n2;i++) {
		for (int j=0;j<n2;j++) {
			int k = n2*i+j;
			if ((i!=0)&(j!=0)) {
				e.back()+=2.*(pow(t[i]*m_hsig_11[k][0]+t[j]*m_hsig_12[k][0],2)+pow(t[i]*m_hsig_12[k][0]+t[j]*m_hsig_22[k][0],2))*(1.+c[i])*(1.+c[j]);
			}
		}
	}
	e.back()/=pow(m_hsig_11[0][0],2)+pow(m_hsig_22[0][0],2)+2.*pow(m_hsig_12[0][0],2);
	e.back()=pow(e.back(),.5);
	cout << "it = " << it << ", e = " << e.back() << endl;
	//
	// Iterative scheme
	while (e.back()>etol) {
		for (int k=0;k<n*n2;k++) {
			m_hdeps_11[k][0]-=hG.r1111[k]*m_hsig_11[k][0]+hG.r1122[k]*m_hsig_22[k][0]+2.*hG.r1112[k]*m_hsig_12[k][0];
			m_hdeps_11[k][1]-=hG.r1111[k]*m_hsig_11[k][1]+hG.r1122[k]*m_hsig_22[k][1]+2.*hG.r1112[k]*m_hsig_12[k][1];
			m_hdeps_22[k][0]-=hG.r1122[k]*m_hsig_11[k][0]+hG.r2222[k]*m_hsig_22[k][0]+2.*hG.r2212[k]*m_hsig_12[k][0];
			m_hdeps_22[k][1]-=hG.r1122[k]*m_hsig_11[k][1]+hG.r2222[k]*m_hsig_22[k][1]+2.*hG.r2212[k]*m_hsig_12[k][1];
			m_hdeps_12[k][0]-=hG.r1112[k]*m_hsig_11[k][0]+hG.r2212[k]*m_hsig_22[k][0]+2.*hG.r1212[k]*m_hsig_12[k][0];
			m_hdeps_12[k][1]-=hG.r1112[k]*m_hsig_11[k][1]+hG.r2212[k]*m_hsig_22[k][1]+2.*hG.r1212[k]*m_hsig_12[k][1];
			// Remark: We do have m_hdeps_ij[0][0|1] == 0
			m_hdeps_11_tmp[k][0]=m_hdeps_11[k][0]; m_hdeps_22_tmp[k][0]=m_hdeps_22[k][0]; m_hdeps_12_tmp[k][0]=m_hdeps_12[k][0];
			m_hdeps_11_tmp[k][1]=m_hdeps_11[k][1]; m_hdeps_22_tmp[k][1]=m_hdeps_22[k][1]; m_hdeps_12_tmp[k][1]=m_hdeps_12[k][1];
		}		
		//
		fftw_execute(ifft_hdeps_11); fftw_execute(ifft_hdeps_22); fftw_execute(ifft_hdeps_12);
		//
		//int k_ind=n2*297+215;
		//int k_ind2=n*297+215;
		//cout << "m_hsig_11 = " << m_hsig_11[k_ind][0]*cte << " , " << m_hsig_11[k_ind][1]*cte << endl;
		//cout << "m_hsig_22 = " << m_hsig_22[k_ind][0]*cte << " , " << m_hsig_22[k_ind][1]*cte << endl;
		//cout << "m_hsig_12 = " << m_hsig_12[k_ind][0]*cte << " , " << m_hsig_12[k_ind][1]*cte << endl;
		//cout << "m_hdeps_11 = " << m_hdeps_11[k_ind][0]*cte << " , " << m_hdeps_11[k_ind][1]*cte << endl;
		//cout << "m_hdeps_22 = " << m_hdeps_22[k_ind][0]*cte << " , " << m_hdeps_22[k_ind][1]*cte << endl;
		//cout << "m_hdeps_12 = " << m_hdeps_12[k_ind][0]*cte << " , " << m_hdeps_12[k_ind][1]*cte << endl;
		//cout << "m_deps_ij = " << m_deps_11[k_ind2]*cte << ", "	<< m_deps_22[k_ind2]*cte << ", " << m_deps_12[k_ind2]*cte << endl;	
		// First passage. Everything is exactly like the Python implementation, except for m_deps_12.
		//
		//cout << "m_hsig_11 = " << m_hsig_11[k_ind][0]*cte << " , " << m_hsig_11[k_ind][1]*cte << endl;
		//cout << "m_hsig_22 = " << m_hsig_22[k_ind][0]*cte << " , " << m_hsig_22[k_ind][1]*cte << endl;
		//cout << "m_hsig_12 = " << m_hsig_12[k_ind][0]*cte << " , " << m_hsig_12[k_ind][1]*cte << endl;
		//
		/*
		double max_deps_11, max_deps_22, max_deps_12;
		for (int k=0;k<n*n;k++) {
			if (m_deps_11[k]>max_deps_11) max_deps_11=cte*m_deps_11[k];
			if (m_deps_22[k]>max_deps_22) max_deps_22=cte*m_deps_22[k];
			if (m_deps_12[k]>max_deps_12) max_deps_12=cte*m_deps_12[k];
		}
		cout << "max deps_ij = " << max_deps_11 << "," <<  max_deps_22 << "," <<  max_deps_12 << endl;				
		*/
		for (int k=0;k<n*n;k++) {
			m_sig_11[k]=m0_sig_11[k]+L[I[k]].r1111*cte*m_deps_11[k]+L[I[k]].r1122*cte*m_deps_22[k]+2.*L[I[k]].r1112*cte*m_deps_12[k];
			m_sig_22[k]=m0_sig_22[k]+L[I[k]].r1122*cte*m_deps_11[k]+L[I[k]].r2222*cte*m_deps_22[k]+2.*L[I[k]].r2212*cte*m_deps_12[k];
			m_sig_12[k]=m0_sig_12[k]+L[I[k]].r1112*cte*m_deps_11[k]+L[I[k]].r2212*cte*m_deps_22[k]+2.*L[I[k]].r1212*cte*m_deps_12[k];
		}
		//
		fftw_execute(fft_sig_11); fftw_execute(fft_sig_22); fftw_execute(fft_sig_12);
		//
		e.push_back((pow(t[0]*m_hsig_11[0][0]+t[0]*m_hsig_12[0][0],2)+pow(t[0]*m_hsig_12[0][0]+t[0]*m_hsig_22[0][0],2))*(1.+c[0])*(1.+c[0]));
		for (int i=0;i<n;i++) {
			for (int j=0;j<n2;j++) {
				int k = n2*i+j;
				if ((i!=0)&(j!=0)) {
					e.back()+=2.*(pow(t[i]*m_hsig_11[k][0]+t[j]*m_hsig_12[k][0],2)+pow(t[i]*m_hsig_12[k][0]+t[j]*m_hsig_22[k][0],2))*(1.+c[i])*(1.+c[j]);
				}
			}
		}
		e.back()/=pow(m_hsig_11[0][0],2)+pow(m_hsig_22[0][0],2)+2.*pow(m_hsig_12[0][0],2);
		e.back()=pow(e.back(),.5);
		cout << "it = " << it << ", e = " << e.back() << endl;
		//
		it+=1;
		//e.back()=pow(10.,-12);
	}
	//
	for (int k=0;k<n*n;k++) {
		m_deps_11[k]=cte*m_deps_11[k]+eps_av_11;
		m_deps_22[k]=cte*m_deps_22[k]+eps_av_22;
		m_deps_12[k]=cte*m_deps_12[k]+eps_av_12;
	}
	//
	sol my_sol;
	my_sol.n=n;
	my_sol.sig_11=m_sig_11; my_sol.sig_22=m_sig_22; my_sol.sig_12=m_sig_12;
	my_sol.eps_11=m_deps_11; my_sol.eps_22=m_deps_22; my_sol.eps_12=m_deps_12;
	
	//int k_ind2=n*297+215;
	//cout << "m_eps_ij = " << m_deps_11[k_ind2] << ", "	<< m_deps_22[k_ind2] << ", " << m_deps_12[k_ind2] << endl;
	//cout << "m_sig_ij = " << m_sig_11[k_ind2] << ", "	<< m_sig_22[k_ind2] << ", " << m_sig_12[k_ind2] << endl;
	
	my_sol.err=e; my_sol.etol=etol;	my_sol.niter=it;
	//
	fftw_destroy_plan(fft_sig_11); fftw_destroy_plan(fft_sig_22); fftw_destroy_plan(fft_sig_12);
	fftw_destroy_plan(ifft_hdeps_11); fftw_destroy_plan(ifft_hdeps_22); fftw_destroy_plan(ifft_hdeps_12);
	//
	return my_sol;
}


//
// Local constitutive models
vector<double> get_stress(material &mat, double eps11_kk, double eps22_kk, double eps12_kk, double dt) {
		double sig11_kk, sig22_kk, sig12_kk;
		//
		if (mat.type==0) {
			sig11_kk=mat.L.r1111*eps11_kk+mat.L.r1122*eps22_kk+2.*mat.L.r1112*eps12_kk;
			sig22_kk=mat.L.r1122*eps11_kk+mat.L.r2222*eps22_kk+2.*mat.L.r2212*eps12_kk;
			sig12_kk=mat.L.r1112*eps11_kk+mat.L.r2212*eps22_kk+2.*mat.L.r1212*eps12_kk;
		}
		else if (mat.type==1) {
			// 
			// type=1: 2D von Mises perfect elastoplastic medium
			/*
			sig11_kk=mat.sig11_k+mat.L.r1111*(eps11_kk-mat.eps11_k)+mat.L.r1122*(eps22_kk-mat.eps22_k)+2.*mat.L.r1112*(eps12_kk-mat.eps12_k);
			sig22_kk=mat.sig22_k+mat.L.r1122*(eps11_kk-mat.eps11_k)+mat.L.r2222*(eps22_kk-mat.eps22_k)+2.*mat.L.r2212*(eps12_kk-mat.eps12_k);
			sig12_kk=mat.sig12_k+mat.L.r1112*(eps11_kk-mat.eps11_k)+mat.L.r2212*(eps22_kk-mat.eps22_k)+2.*mat.L.r1212*(eps12_kk-mat.eps12_k);
			double tr_sig_kk=sig11_kk+sig22_kk;
			double s11_kk=sig11_kk-.5*tr_sig_kk;
			double s22_kk=sig22_kk-.5*tr_sig_kk;
			double s12_kk=sig12_kk;
			double sig_eq_kk=pow(3./2.*(s11_kk*s11_kk+s22_kk*s22_kk+2.*s12_kk*s12_kk),.5);
			mat.plastified=false;
			if (sig_eq_kk>=mat.sig0) {
				double dp=(sig_eq_kk-mat.sig0)/3./mat.mu2d;
				sig11_kk-=3.*mat.mu2d*dp*s11_kk/sig_eq_kk;
				sig22_kk-=3.*mat.mu2d*dp*s22_kk/sig_eq_kk;
				sig12_kk-=3.*mat.mu2d*dp*s12_kk/sig_eq_kk;
				mat.p+=dp;
			}
			*/
			sig11_kk=mat.sig11_k+mat.L.r1111*(eps11_kk-mat.eps11_k)+mat.L.r1122*(eps22_kk-mat.eps22_k)+2.*mat.L.r1112*(eps12_kk-mat.eps12_k);
			sig22_kk=mat.sig22_k+mat.L.r1122*(eps11_kk-mat.eps11_k)+mat.L.r2222*(eps22_kk-mat.eps22_k)+2.*mat.L.r2212*(eps12_kk-mat.eps12_k);
			sig12_kk=mat.sig12_k+mat.L.r1112*(eps11_kk-mat.eps11_k)+mat.L.r2212*(eps22_kk-mat.eps22_k)+2.*mat.L.r1212*(eps12_kk-mat.eps12_k);
			double tr_sig_kk=sig11_kk+sig22_kk;
			double s11_kk=sig11_kk-.5*tr_sig_kk;
			double s22_kk=sig22_kk-.5*tr_sig_kk;
			double s12_kk=sig12_kk;
			double sij_norm=pow((s11_kk*s11_kk+s22_kk*s22_kk+2.*s12_kk*s12_kk),.5);
			mat.plastified=false;
			if (sij_norm>=mat.sig0) {
				//double dp=(sig_eq_kk-mat.sig0)/3./mat.mu2d;
				sig11_kk=.5*tr_sig_kk+mat.sig0*s11_kk/sij_norm;
				sig22_kk=.5*tr_sig_kk+mat.sig0*s22_kk/sij_norm;
				sig12_kk=mat.sig0*s12_kk/sij_norm;
				mat.plastified=true;	
			}		
		}
		else if (mat.type==3) {
			sig11_kk=mat.sig11_k+mat.L.r1111*(eps11_kk-mat.eps11_k)+mat.L.r1122*(eps22_kk-mat.eps22_k)+2.*mat.L.r1112*(eps12_kk-mat.eps12_k);
			sig22_kk=mat.sig22_k+mat.L.r1122*(eps11_kk-mat.eps11_k)+mat.L.r2222*(eps22_kk-mat.eps22_k)+2.*mat.L.r2212*(eps12_kk-mat.eps12_k);
			sig12_kk=mat.sig12_k+mat.L.r1112*(eps11_kk-mat.eps11_k)+mat.L.r2212*(eps22_kk-mat.eps22_k)+2.*mat.L.r1212*(eps12_kk-mat.eps12_k);
			double tr_sig_kk=sig11_kk+sig22_kk;
			double s11_kk=sig11_kk-.5*tr_sig_kk;
			double s22_kk=sig22_kk-.5*tr_sig_kk;
			double s12_kk=sig12_kk;
			double sij_norm=pow((s11_kk*s11_kk+s22_kk*s22_kk+2.*s12_kk*s12_kk),.5);
			mat.plastified=false;
			if (sij_norm>=mat.sig0) {
				double dp=0;
				double phi=pow((sij_norm-2.*mat.mu2d*dp-mat.sig0)/mat.k,1./mat.m);
				double residual=phi;
				while (abs(residual)>pow(10.,-9)) {
					double phi=pow((sij_norm-2.*mat.mu2d*dp-mat.sig0)/mat.k,1./mat.m);
					dp+=(dt*phi-dp)/(1.+dt*2.*mat.mu2d/mat.m/mat.k*pow(phi,1.-mat.m));
					residual=dp-phi;
				}
				sig11_kk=.5*tr_sig_kk+(mat.sig0+mat.k*pow(dp/dt,mat.m))*s11_kk/sij_norm;
				sig22_kk=.5*tr_sig_kk+(mat.sig0+mat.k*pow(dp/dt,mat.m))*s22_kk/sij_norm;
				sig12_kk=(mat.sig0+mat.k*pow(dp/dt,mat.m))*s12_kk/sij_norm;
				mat.plastified=true;	
			}
		}
		else if (mat.type==4) {
			sig11_kk=mat.sig11_k+mat.L.r1111*(eps11_kk-mat.eps11_k)+mat.L.r1122*(eps22_kk-mat.eps22_k)+2.*mat.L.r1112*(eps12_kk-mat.eps12_k);
			sig22_kk=mat.sig22_k+mat.L.r1122*(eps11_kk-mat.eps11_k)+mat.L.r2222*(eps22_kk-mat.eps22_k)+2.*mat.L.r2212*(eps12_kk-mat.eps12_k);
			sig12_kk=mat.sig12_k+mat.L.r1112*(eps11_kk-mat.eps11_k)+mat.L.r2212*(eps22_kk-mat.eps22_k)+2.*mat.L.r1212*(eps12_kk-mat.eps12_k);
			double tr_sig_kk=sig11_kk+sig22_kk;
			double s11_kk=sig11_kk-.5*tr_sig_kk;
			double s22_kk=sig22_kk-.5*tr_sig_kk;
			double s12_kk=sig12_kk;
			double sij_norm=pow((s11_kk*s11_kk+s22_kk*s22_kk+2.*s12_kk*s12_kk),.5);
			mat.plastified=false;
			mat.dp=0;
			if (sij_norm>=mat.sig0) {
				double dp=0;
				double residual=dp-dt/mat.tau*(pow((sij_norm-2.*mat.mu2d*dp)/mat.sig0,1./mat.e)-1.);
				while (abs(residual)>pow(10.,-7)) {
					dp-=(dp-dt/mat.tau*(pow((sij_norm-2.*mat.mu2d*dp)/mat.sig0,1./mat.e)-1.))/(1.+2.*mat.mu2d*dt/mat.e/mat.tau/mat.sig0*pow((sij_norm-2.*mat.mu2d*dp)/mat.sig0,1./mat.e-1.));
					residual=dp-dt/mat.tau*(pow((sij_norm-2.*mat.mu2d*dp)/mat.sig0,1./mat.e)-1.);
				}
				mat.dp=dp;
				sig11_kk=.5*tr_sig_kk+mat.sig0*pow(1.+mat.tau*dp/dt,mat.e)*s11_kk/sij_norm;
				sig22_kk=.5*tr_sig_kk+mat.sig0*pow(1.+mat.tau*dp/dt,mat.e)*s22_kk/sij_norm;
				sig12_kk=mat.sig0*pow(1.+mat.tau*dp/dt,mat.e)*s12_kk/sij_norm;
				mat.plastified=true;	
			}
		}
		else if (mat.type==5) {			
			double sig11_0kk=mat.sig11_k+mat.L.r1111*(eps11_kk-mat.eps11_k)+mat.L.r1122*(eps22_kk-mat.eps22_k)+2.*mat.L.r1112*(eps12_kk-mat.eps12_k);
			double sig22_0kk=mat.sig22_k+mat.L.r1122*(eps11_kk-mat.eps11_k)+mat.L.r2222*(eps22_kk-mat.eps22_k)+2.*mat.L.r2212*(eps12_kk-mat.eps12_k);
			double sig12_0kk=mat.sig12_k+mat.L.r1112*(eps11_kk-mat.eps11_k)+mat.L.r2212*(eps22_kk-mat.eps22_k)+2.*mat.L.r1212*(eps12_kk-mat.eps12_k);
			sig11_kk=sig11_0kk;
			sig22_kk=sig22_0kk;			
			sig12_kk=sig12_0kk;
			//
			vector<double> t0=mat.t0;
			//
			vector<double> tau(mat.nslip);
			vector<double> Dgam(mat.nslip);
			vector<double> cte_b(mat.nslip);
			//
			double psi11=sig11_kk-sig11_0kk;
			double psi22=sig22_kk-sig22_0kk;
			double psi12=sig12_kk-sig12_0kk;			
			
			
//			cout << "t0, " << t0[0] << "," << t0[1] << endl;
			
			for (int al=0;al<mat.nslip;al++) {
				tau[al]=sig11_kk*mat.m11[al]+sig22_kk*mat.m22[al]+2.*sig12_kk*mat.m12[al];
				//
				Dgam[al]=dt*mat.dgam0[al]*pow(abs(tau[al]/t0[al]),1./mat.m)*sgn(tau[al]);
				cte_b[al]=dt*mat.dgam0[al]/mat.m/t0[al]*pow(abs(tau[al]/t0[al]),1./mat.m-1.);
				//
				psi11+=Dgam[al]*mat.C11rs_mrs[al];
				psi22+=Dgam[al]*mat.C22rs_mrs[al];
				psi12+=Dgam[al]*mat.C12rs_mrs[al];
			}
			/*
			cout << "mat.th[0]=" << mat.th[0] << ", mst.th[1]=" << mat.th[1] << endl;
			cout << "mat.m11[0]=" << mat.m11[0] << ", mst.m22[0]=" << mat.m22[0] << endl;
			cout << mat.C11rs_mrs_m11[0] << ", " << mat.C11rs_mrs_m22[0] << ", " << mat.C11rs_mrs_m12[0] << endl;
			cout << mat.C22rs_mrs_m11[0] << ", " << mat.C22rs_mrs_m22[0] << ", " << mat.C22rs_mrs_m12[0] << endl;
			cout << mat.C12rs_mrs_m11[0] << ", " << mat.C12rs_mrs_m22[0] << ", " << mat.C12rs_mrs_m12[0] << endl;
			cout << endl;
//			cout << "psi_ij" << psi11 << "," << psi22 << "," << psi12 << endl;			
			*/
			int iter=0;
			while (pow(psi11*psi11+psi22*psi22+psi12*psi12,.5)>pow(10.,-9.)) {
				while (pow(psi11*psi11+psi22*psi22+psi12*psi12,.5)>pow(10.,-9.)) {
					double J1111=1., J1122=0., J1112=0.;
					double J2211=0., J2222=1., J2212=0.;
					double J1211=0., J1222=0., J1212=.5;
//					cout << "cte_b, " << cte_b[0] << "," << cte_b[1] << endl;
					for (int al=0;al<mat.nslip;al++) {
						J1111+=cte_b[al]*mat.C11rs_mrs_m11[al];
						J1122+=cte_b[al]*mat.C11rs_mrs_m22[al];
						J1112+=cte_b[al]*mat.C11rs_mrs_m12[al];
						J2211+=cte_b[al]*mat.C22rs_mrs_m11[al];
						J2222+=cte_b[al]*mat.C22rs_mrs_m22[al];
						J2212+=cte_b[al]*mat.C22rs_mrs_m12[al];
						J1211+=cte_b[al]*mat.C12rs_mrs_m11[al];
						J1222+=cte_b[al]*mat.C12rs_mrs_m22[al];
						J1212+=cte_b[al]*mat.C12rs_mrs_m12[al];
					}

					vector<double> Jinv=inv_mat_3by3({J1111,J1122,J1112*pow(2.,.5),J2211,J2222,J2212*pow(2.,-.5),J1211*pow(2.,-.5),J1222*pow(2.,-.5),2.*J1212});
					double Jinv1111=Jinv[0], Jinv1122=Jinv[1], Jinv1112=Jinv[2];
					double Jinv2211=Jinv[3], Jinv2222=Jinv[4], Jinv2212=Jinv[5];
					double Jinv1211=Jinv[6], Jinv1222=Jinv[7], Jinv1212=Jinv[8];					
					Jinv1112*=pow(2.,-.5);
					Jinv2212*=pow(2.,-.5);
					Jinv1211*=pow(2.,-.5);	
					Jinv1222*=pow(2.,-.5);	
					Jinv1212*=pow(2.,-.5);	
//					cout << Jinv1111 << "," << Jinv1122 << "," << Jinv1112 << "," << Jinv2211 << "," << Jinv2222 << "," << Jinv2212 << "," << Jinv1211 << "," << Jinv1222 << "," << Jinv1212 << endl;
					//
					sig11_kk-=(Jinv1111*psi11+Jinv1122*psi22+2.*Jinv1112*psi12);
					sig22_kk-=(Jinv2211*psi11+Jinv2222*psi22+2.*Jinv2212*psi12);
					sig12_kk-=(Jinv1211*psi11+Jinv1222*psi22+2.*Jinv1212*psi12);
					//
					psi11=sig11_kk-sig11_0kk;
					psi22=sig22_kk-sig22_0kk;
					psi12=sig12_kk-sig12_0kk;			
					for (int al=0;al<mat.nslip;al++) {
						tau[al]=sig11_kk*mat.m11[al]+sig22_kk*mat.m22[al]+2.*sig12_kk*mat.m12[al];
						//
						Dgam[al]=dt*mat.dgam0[al]*pow(abs(tau[al]/t0[al]),1./mat.m)*sgn(tau[al]);
						cte_b[al]=dt*mat.dgam0[al]/mat.m/t0[al]*pow(abs(tau[al]/t0[al]),1./mat.m-1.);	
						//
						psi11+=Dgam[al]*mat.C11rs_mrs[al];
						psi22+=Dgam[al]*mat.C22rs_mrs[al];
						psi12+=Dgam[al]*mat.C12rs_mrs[al];
					}
					iter+=1;				
				}
				psi11=sig11_kk-sig11_0kk;
				psi22=sig22_kk-sig22_0kk;
				psi12=sig12_kk-sig12_0kk;
				for (int al=0;al<mat.nslip;al++) {
					//tau[al]=sig11_kk*mat.m11[al]+sig22_kk*mat.m22[al]+2.*sig12_kk*mat.m12[al];
					//
					t0[al]=mat.t0[al]+mat.h*abs(Dgam[al]);
					Dgam[al]=dt*mat.dgam0[al]*pow(abs(tau[al]/t0[al]),1./mat.m)*sgn(tau[al]);
					cte_b[al]=dt*mat.dgam0[al]/mat.m/t0[al]*pow(abs(tau[al]/t0[al]),1./mat.m-1.);
					//
					psi11+=Dgam[al]*mat.C11rs_mrs[al];
					psi22+=Dgam[al]*mat.C22rs_mrs[al];
					psi12+=Dgam[al]*mat.C12rs_mrs[al];
				}	
			}
		mat.dp=0;
		for (int al=0;al<mat.nslip;al++) {
			mat.Dgam[al]=Dgam[al];
			mat.dp+=abs(Dgam[al]);
		}	
		}
		
		
	return {sig11_kk,sig22_kk,sig12_kk};
}

//
// Direct solver
sol nonlinear_direct_solver(int n, T4_field hG, vector<int> I, vector<material> mat, vector<double> eps_av_11, vector<double> eps_av_22, vector<double> eps_av_12, vector<double> time, double etol, string proj_name) {
	// 		   n: Number of frequencies in each direction -- must be even.
	// 		   L: Stiffness tensor field.
	// eps_av_11: Average axial strain prescribed along e_1.
	// eps_av_22: Average axial strain prescribed along e_2.
	// eps_av_12: Average shear strain prescribed along e_1 and e_2.
	// 		etol: Tolerance for the error in equlibrium.
	//
	int n2 = n/2+1;
	double cte = 1./(n*n);
	//vector<double> m0_sig_11(n*n), m0_sig_22(n*n), m0_sig_12(n*n);
	double* m_sig_11 = static_cast<double*>(fftw_malloc(n*n*sizeof(double)));
	double* m_sig_22 = static_cast<double*>(fftw_malloc(n*n*sizeof(double)));
	double* m_sig_12 = static_cast<double*>(fftw_malloc(n*n*sizeof(double)));
	double* m_deps_11 = static_cast<double*>(fftw_malloc(n*n*sizeof(double)));
	double* m_deps_22 = static_cast<double*>(fftw_malloc(n*n*sizeof(double)));
	double* m_deps_12 = static_cast<double*>(fftw_malloc(n*n*sizeof(double)));
	double* m_eps_11 = static_cast<double*>(fftw_malloc(n*n*sizeof(double)));
	double* m_eps_22 = static_cast<double*>(fftw_malloc(n*n*sizeof(double)));
	double* m_eps_12 = static_cast<double*>(fftw_malloc(n*n*sizeof(double)));
	//vector<vector<double>> m_hdeps_11(n*n2,vector<double>(2)), m_hdeps_22(n*n2,vector<double>(2)), m_hdeps_12(n*n2,vector<double>(2));

	fftw_complex* m_hdeps_11 = static_cast<fftw_complex*>(fftw_malloc(n*n2*sizeof(fftw_complex)));
	fftw_complex* m_hdeps_22 = static_cast<fftw_complex*>(fftw_malloc(n*n2*sizeof(fftw_complex)));
	fftw_complex* m_hdeps_12 = static_cast<fftw_complex*>(fftw_malloc(n*n2*sizeof(fftw_complex)));

	fftw_complex* m_hsig_11 = static_cast<fftw_complex*>(fftw_malloc(n*n2*sizeof(fftw_complex)));
	fftw_complex* m_hsig_22 = static_cast<fftw_complex*>(fftw_malloc(n*n2*sizeof(fftw_complex)));
	fftw_complex* m_hsig_12 = static_cast<fftw_complex*>(fftw_malloc(n*n2*sizeof(fftw_complex)));
	fftw_complex* m_hdeps_11_tmp = static_cast<fftw_complex*>(fftw_malloc(n*n2*sizeof(fftw_complex)));
	fftw_complex* m_hdeps_22_tmp = static_cast<fftw_complex*>(fftw_malloc(n*n2*sizeof(fftw_complex)));
	fftw_complex* m_hdeps_12_tmp = static_cast<fftw_complex*>(fftw_malloc(n*n2*sizeof(fftw_complex)));
	//
	// Wave vectors
	double q;
	vector<double> t, c;
	q = 0.;
	t.push_back(tan(q/2.));
	c.push_back(cos(q));
	for (int i=1;i<n/2+1;i++) {
		q =2.*M_PI/n*i;
		t.push_back(tan(q/2.));
		c.push_back(cos(q));
	}
	for (int i=-n/2+1;i<0;i++) {
		q= 2.*M_PI/n*i;
		t.push_back(tan(q/2.));
		c.push_back(cos(q));
	}
	//
	// Set up FFTW plans
	int flags1 = FFTW_ESTIMATE;
	int flags2 = FFTW_ESTIMATE | FFTW_BACKWARD;
	fftw_plan fft_sig_11 = fftw_plan_dft_r2c_2d(n,n,m_sig_11,m_hsig_11,flags1);
	fftw_plan fft_sig_22 = fftw_plan_dft_r2c_2d(n,n,m_sig_22,m_hsig_22,flags1);
	fftw_plan fft_sig_12 = fftw_plan_dft_r2c_2d(n,n,m_sig_12,m_hsig_12,flags1);
	fftw_plan ifft_hdeps_11 = fftw_plan_dft_c2r_2d(n,n,m_hdeps_11_tmp,m_deps_11,flags2);
	fftw_plan ifft_hdeps_22 = fftw_plan_dft_c2r_2d(n,n,m_hdeps_22_tmp,m_deps_22,flags2);
	fftw_plan ifft_hdeps_12 = fftw_plan_dft_c2r_2d(n,n,m_hdeps_12_tmp,m_deps_12,flags2);
	//
	for (int k=0;k<n*n;k++) {
		m_deps_11[k]=0; m_eps_11[k]=0; m_sig_11[k]=0;
		m_deps_22[k]=0; m_eps_22[k]=0; m_sig_22[k]=0;
		m_deps_12[k]=0; m_eps_12[k]=0; m_sig_12[k]=0;
	}
	//
	int nt=eps_av_11.size();
	for (int tk=1;tk<nt;tk++) {
		//
		// Initiliaze 
		for (int k=0;k<n*n;k++) {
			if (k<n*n2) {
				m_hdeps_11[k][0]=0.; m_hdeps_11[k][1]=0.;
				m_hdeps_22[k][0]=0.; m_hdeps_22[k][1]=0.;
				m_hdeps_12[k][0]=0.; m_hdeps_12[k][1]=0.;
				
			}
			mat[k].eps11_k=m_eps_11[k]; mat[k].sig11_k=m_sig_11[k];
			mat[k].eps22_k=m_eps_22[k]; mat[k].sig22_k=m_sig_22[k];
			mat[k].eps12_k=m_eps_12[k]; mat[k].sig12_k=m_sig_12[k];
			//
			// update internal variables
			mat[k].p+=mat[k].dp;
			if (mat[k].type==5) {
				for (int al=0;al<mat[k].nslip;al++) {
					//mat[k].gam[al]+=mat[k].Dgam[al];
					mat[k].t0[al]+=mat[k].h*abs(mat[k].Dgam[al]);
				}
			}
			m_eps_11[k]=eps_av_11[tk]+cte*m_deps_11[k];
			m_eps_22[k]=eps_av_22[tk]+cte*m_deps_22[k];
			m_eps_12[k]=eps_av_12[tk]+cte*m_deps_12[k];
			vector<double> sig_ij=get_stress(mat[k],m_eps_11[k],m_eps_22[k],m_eps_12[k],time[tk]-time[tk-1]);
			m_sig_11[k]=sig_ij[0];
			m_sig_22[k]=sig_ij[1];
			m_sig_12[k]=sig_ij[2];
			
			//cout << I[k] << ", mat[k].type = " << mat[k].type << endl; 
			//cout << m_eps_11[k] << ", " << m_eps_22[k] << ", " << m_eps_12[k] << ", " << sig_ij[0] << ", " << sig_ij[0] << ", " << sig_ij[0] << endl;
			
			//cout << "test2" << endl;
			
		}
		//
		fftw_execute(fft_sig_11); fftw_execute(fft_sig_22); fftw_execute(fft_sig_12);
		//	
		vector<double> e;
		e.push_back((pow(t[0]*m_hsig_11[0][0]+t[0]*m_hsig_12[0][0],2)+pow(t[0]*m_hsig_12[0][0]+t[0]*m_hsig_22[0][0],2))*(1.+c[0])*(1.+c[0]));
		for (int i=0;i<n2;i++) {
			for (int j=0;j<n2;j++) {
				int k = n2*i+j;
				if ((i!=0)&(j!=0)) {
					e.back()+=2.*(pow(t[i]*m_hsig_11[k][0]+t[j]*m_hsig_12[k][0],2)+pow(t[i]*m_hsig_12[k][0]+t[j]*m_hsig_22[k][0],2))*(1.+c[i])*(1.+c[j]);
				}
			}
		}
		e.back()/=pow(m_hsig_11[0][0],2)+pow(m_hsig_22[0][0],2)+2.*pow(m_hsig_12[0][0],2);
		e.back()=pow(e.back(),.5);
		int it = 0;
		cout << "tk = " << tk << ", it = " << it << ", e = " << e.back() << endl;
		//
		// Iterative scheme
		while (e.back()>etol) {
			for (int k=0;k<n*n2;k++) {
				m_hdeps_11[k][0]-=hG.r1111[k]*m_hsig_11[k][0]+hG.r1122[k]*m_hsig_22[k][0]+2.*hG.r1112[k]*m_hsig_12[k][0];
				m_hdeps_11[k][1]-=hG.r1111[k]*m_hsig_11[k][1]+hG.r1122[k]*m_hsig_22[k][1]+2.*hG.r1112[k]*m_hsig_12[k][1];
				m_hdeps_22[k][0]-=hG.r1122[k]*m_hsig_11[k][0]+hG.r2222[k]*m_hsig_22[k][0]+2.*hG.r2212[k]*m_hsig_12[k][0];
				m_hdeps_22[k][1]-=hG.r1122[k]*m_hsig_11[k][1]+hG.r2222[k]*m_hsig_22[k][1]+2.*hG.r2212[k]*m_hsig_12[k][1];
				m_hdeps_12[k][0]-=hG.r1112[k]*m_hsig_11[k][0]+hG.r2212[k]*m_hsig_22[k][0]+2.*hG.r1212[k]*m_hsig_12[k][0];
				m_hdeps_12[k][1]-=hG.r1112[k]*m_hsig_11[k][1]+hG.r2212[k]*m_hsig_22[k][1]+2.*hG.r1212[k]*m_hsig_12[k][1];
				// Remark: We do have m_hdeps_ij[0][0|1] == 0
				m_hdeps_11_tmp[k][0]=m_hdeps_11[k][0]; m_hdeps_22_tmp[k][0]=m_hdeps_22[k][0]; m_hdeps_12_tmp[k][0]=m_hdeps_12[k][0];
				m_hdeps_11_tmp[k][1]=m_hdeps_11[k][1]; m_hdeps_22_tmp[k][1]=m_hdeps_22[k][1]; m_hdeps_12_tmp[k][1]=m_hdeps_12[k][1];
			}		
			//
			fftw_execute(ifft_hdeps_11); fftw_execute(ifft_hdeps_22); fftw_execute(ifft_hdeps_12);
			//
			//int k_ind=n2*297+215;
			//int k_ind2=n*297+215;
			//cout << "m_hsig_11 = " << m_hsig_11[k_ind][0]*cte << " , " << m_hsig_11[k_ind][1]*cte << endl;
			//cout << "m_hsig_22 = " << m_hsig_22[k_ind][0]*cte << " , " << m_hsig_22[k_ind][1]*cte << endl;
			//cout << "m_hsig_12 = " << m_hsig_12[k_ind][0]*cte << " , " << m_hsig_12[k_ind][1]*cte << endl;
			//cout << "m_hdeps_11 = " << m_hdeps_11[k_ind][0]*cte << " , " << m_hdeps_11[k_ind][1]*cte << endl;
			//cout << "m_hdeps_22 = " << m_hdeps_22[k_ind][0]*cte << " , " << m_hdeps_22[k_ind][1]*cte << endl;
			//cout << "m_hdeps_12 = " << m_hdeps_12[k_ind][0]*cte << " , " << m_hdeps_12[k_ind][1]*cte << endl;
			//cout << "m_deps_ij = " << m_deps_11[k_ind2]*cte << ", "	<< m_deps_22[k_ind2]*cte << ", " << m_deps_12[k_ind2]*cte << endl;	
			// First passage. Everything is exactly like the Python implementation, except for m_deps_12.
			//
			//cout << "m_hsig_11 = " << m_hsig_11[k_ind][0]*cte << " , " << m_hsig_11[k_ind][1]*cte << endl;
			//cout << "m_hsig_22 = " << m_hsig_22[k_ind][0]*cte << " , " << m_hsig_22[k_ind][1]*cte << endl;
			//cout << "m_hsig_12 = " << m_hsig_12[k_ind][0]*cte << " , " << m_hsig_12[k_ind][1]*cte << endl;
			//
			/*
			double max_deps_11, max_deps_22, max_deps_12;
			for (int k=0;k<n*n;k++) {
				if (m_deps_11[k]>max_deps_11) max_deps_11=cte*m_deps_11[k];
				if (m_deps_22[k]>max_deps_22) max_deps_22=cte*m_deps_22[k];
				if (m_deps_12[k]>max_deps_12) max_deps_12=cte*m_deps_12[k];
			}
			cout << "max deps_ij = " << max_deps_11 << "," <<  max_deps_22 << "," <<  max_deps_12 << endl;				
			*/
			//
			for (int k=0;k<n*n;k++) {
				m_eps_11[k]=eps_av_11[tk]+cte*m_deps_11[k];
				m_eps_22[k]=eps_av_22[tk]+cte*m_deps_22[k];
				m_eps_12[k]=eps_av_12[tk]+cte*m_deps_12[k];
				vector<double> sig_ij=get_stress(mat[k],m_eps_11[k],m_eps_22[k],m_eps_12[k],time[tk]-time[tk-1]);
				m_sig_11[k]=sig_ij[0];
				m_sig_22[k]=sig_ij[1];
				m_sig_12[k]=sig_ij[2];
			}
			//
			fftw_execute(fft_sig_11); fftw_execute(fft_sig_22); fftw_execute(fft_sig_12);
			//
			e.push_back((pow(t[0]*m_hsig_11[0][0]+t[0]*m_hsig_12[0][0],2)+pow(t[0]*m_hsig_12[0][0]+t[0]*m_hsig_22[0][0],2))*(1.+c[0])*(1.+c[0]));
			for (int i=0;i<n;i++) {
				for (int j=0;j<n2;j++) {
					int k = n2*i+j;
					if ((i!=0)&(j!=0)) {
						e.back()+=2.*(pow(t[i]*m_hsig_11[k][0]+t[j]*m_hsig_12[k][0],2)+pow(t[i]*m_hsig_12[k][0]+t[j]*m_hsig_22[k][0],2))*(1.+c[i])*(1.+c[j]);
					}
				}
			}
			e.back()/=pow(m_hsig_11[0][0],2)+pow(m_hsig_22[0][0],2)+2.*pow(m_hsig_12[0][0],2);
			e.back()=pow(e.back(),.5);
			cout << "tk = " << tk << ", it = " << it << ", e = " << e.back() << endl;
			//
			it+=1;
			//e.back()=pow(10.,-12);
		}
		ostringstream num;
		num << tk;	
		write_output(m_sig_11,m_sig_22,m_sig_12,n,proj_name+".sig_ij_"+num.str());
		write_output(m_eps_11,m_eps_22,m_eps_12,n,proj_name+".eps_ij_"+num.str());
		//write_p_output(n,mat,proj_name+".p_"+num.str());
		write_void_and_plastified_output(n,mat,proj_name+".plast_"+num.str());
	}
	//
	sol my_sol;
	my_sol.n=n;
	my_sol.sig_11=m_sig_11; my_sol.sig_22=m_sig_22; my_sol.sig_12=m_sig_12;
	my_sol.eps_11=m_eps_11; my_sol.eps_22=m_eps_22; my_sol.eps_12=m_eps_12;
	//my_sol.err=e; my_sol.etol=etol;	my_sol.niter=it;	
	//
	fftw_destroy_plan(fft_sig_11); fftw_destroy_plan(fft_sig_22); fftw_destroy_plan(fft_sig_12);
	fftw_destroy_plan(ifft_hdeps_11); fftw_destroy_plan(ifft_hdeps_22); fftw_destroy_plan(ifft_hdeps_12);
	//
	return my_sol;
}


/*
int main(int argc, char* argv[] ) {
	int n=atoi(argv[1]);			// Number of frequencies in each direction
	T4 L0;
	L0.r1111=atof(argv[2]); 		// L1111 stiffness component
	L0.r1122=atof(argv[3]); 		// L1122 stiffness component
	L0.r1112=atof(argv[4]); 		// L1112 stiffness component
	L0.r2222=atof(argv[5]); 		// L2222 stiffness component
	L0.r2212=atof(argv[6]); 		// L2212 stiffness component
	L0.r1212=atof(argv[7]); 		// L1212 stiffness component	
	float eps11=atof(argv[8]);		// Average axial strain along e1
	float eps22=atof(argv[9]);		// Average axial strain along e2
	float eps12=atof(argv[10]);		// Average shear strain alons e2
	float etol=atof(argv[11]);		// Tolerance for the error in equilibrium
	string geo_fname=argv[12]; 		// Geometry file name
	string mat_fname=argv[13]; 		// Material file name
	string proj_name=argv[14]; 		// Project name
	//
	vector<T4> L=read_mat(mat_fname);	
	vector<int> geo=read_geo(geo_fname);	
	T4_field hGp = set_discrete_green_operator(n, L0);
	//
	sol test = direct_solver(n, hGp, geo, L, eps11, eps22, eps12, etol);
	//
	write_output(test.sig_11,test.sig_22,test.sig_12,n,proj_name+".sig_ij");
	write_output(test.eps_11,test.eps_22,test.eps_12,n,proj_name+".eps_ij");
	write_output(test.err,test.niter,proj_name+".err");		
	//
	return 0;
}
*/


///*
int main(int argc, char* argv[] ) {
	int n=atoi(argv[1]);			// Number of frequencies in each direction
	T4 L0;
	L0.r1111=atof(argv[2]); 		// L1111 stiffness component
	L0.r1122=atof(argv[3]); 		// L1122 stiffness component
	L0.r1112=atof(argv[4]); 		// L1112 stiffness component
	L0.r2222=atof(argv[5]); 		// L2222 stiffness component
	L0.r2212=atof(argv[6]); 		// L2212 stiffness component
	L0.r1212=atof(argv[7]); 		// L1212 stiffness component	
	float etol=atof(argv[8]);		// Tolerance for the error in equilibrium
	string eps_av_fname=argv[9]; 	// Mean strain file name
	string geo_fname=argv[10]; 		// Geometry file name
	string mat_fname=argv[11]; 		// Material file name
	string proj_name=argv[12]; 		// Project name
	//
	//vector<T4> L=read_mat(mat_fname);	
	vector<int> geo=read_geo(geo_fname);	
	T4_field hGp = set_discrete_green_operator(n, L0);
	vector<vector<double>> eps_av=read_eps_av(eps_av_fname);
	vector<double> eps11=eps_av[0];
	vector<double> eps22=eps_av[1];
	vector<double> eps12=eps_av[2];
	vector<double> time=eps_av[3];
	//
	ifstream myfile;
	myfile.open(mat_fname);
	string line;
    getline(myfile,line);
    int nmat=atoi(line.c_str());
	vector<string> mat_strings(nmat);
	for (int k=0;k<nmat;k++) {
		string kline;
		getline(myfile,kline);
		mat_strings[k]=kline;
	}
	myfile.close();	
	//
	vector<material> mat(n*n);
	for (int k=0;k<n*n;k++) {
		ini_mat(mat[k], mat_strings[geo[k]]);
	}
	myfile.close();
	//
	
	//cout << mat[0].t0[0] << "," << mat[0].dgam0[0] << "," << mat[0].L.r1212 << "," << mat[0].h << "," << mat[0].th[0] << "," << mat[0].m << endl;
	
	sol test = nonlinear_direct_solver(n, hGp, geo, mat, eps11, eps22, eps12, time, etol, proj_name);
	//
	write_output(test.sig_11,test.sig_22,test.sig_12,n,proj_name+".sig_ij");
	write_output(test.eps_11,test.eps_22,test.eps_12,n,proj_name+".eps_ij");
	//write_output(test.err,test.niter,proj_name+".err");		
	

	
	//
	return 0;
}
//*/

