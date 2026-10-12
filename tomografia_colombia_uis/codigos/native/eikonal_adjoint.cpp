// Exact reverse differentiation of the converged first-order Godunov stencil.
// Original implementation; graph edges strictly decrease the travel-time field.
#include <vector>
#include <queue>
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <stdexcept>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif
struct Graph {
    int64_t n, nc, source;
    std::vector<double> time, slow, local, coeff, lambda;
    std::vector<int64_t> parent, mapping;
    std::vector<uint64_t> stamp;
    uint64_t epoch=0;
    double max_relative_residual=0;
    Graph(const double* t,const double* s,const int64_t* map,int64_t nz,int64_t ny,int64_t nx,int64_t src,double h,int64_t coarse):
        n(nz*ny*nx),nc(coarse),source(src),time(t,t+n),slow(s,s+n),local(n),coeff(n*3),lambda(n),parent(n*3,-1),mapping(map,map+n),stamp(n) {
        if(src<0 || src>=n || time[src]!=0 || h<=0)throw std::runtime_error("source or spacing");
        for(int64_t u=0;u<n;u++) {
            if(!std::isfinite(time[u]) || slow[u]<=0 || !std::isfinite(slow[u]) || mapping[u]<0 || mapping[u]>=nc)throw std::runtime_error("invalid input");
            if(u==src)continue;
            int64_t i=u%nx,j=(u/nx)%ny,k=u/(nx*ny);
            int64_t neighbors[6]={i>0?u-1:-1,i+1<nx?u+1:-1,j>0?u-nx:-1,j+1<ny?u+nx:-1,k>0?u-nx*ny:-1,k+1<nz?u+nx*ny:-1};
            double den=0,eq=0;
            for(int a=0;a<3;a++) {
                auto v=neighbors[2*a],w=neighbors[2*a+1];
                if(v<0 || (w>=0 && time[w]<time[v]))v=w;
                if(v>=0 && time[v]<time[u]) {
                    double diff=time[u]-time[v];
                    parent[3*u+a]=v;coeff[3*u+a]=diff/(h*h);den+=coeff[3*u+a];eq+=diff*diff/(h*h);
                }
            }
            if(den<=0)throw std::runtime_error("noncausal field");
            max_relative_residual=std::max(max_relative_residual,std::abs(std::sqrt(eq)/slow[u]-1));
            local[u]=slow[u]/den;
            for(int a=0;a<3;a++)coeff[3*u+a]/=den;
        }
        if(max_relative_residual>2e-7)throw std::runtime_error("unconverged Godunov field");
    }
    void row(int64_t start,double* out,double& moment,int64_t& visits) {
        if(start<0 || start>=n)throw std::runtime_error("endpoint outside grid");
        std::fill(out,out+nc,0);moment=0;visits=0;epoch++;
        std::priority_queue<std::pair<double,int64_t>> queue;
        stamp[start]=epoch;lambda[start]=1;queue.emplace(time[start],start);
        while(!queue.empty()) {
            auto u=queue.top().second;queue.pop();visits++;
            if(u==source)continue;
            double l=lambda[u],g=l*local[u];out[mapping[u]]+=g;moment+=g*slow[u];
            for(int a=0;a<3;a++) {
                auto v=parent[3*u+a];if(v<0)continue;
                if(stamp[v]!=epoch){stamp[v]=epoch;lambda[v]=0;queue.emplace(time[v],v);}
                lambda[v]+=l*coeff[3*u+a];
            }
        }
    }
};
API void* adjoint_create(const double* t,const double* s,const int64_t* map,int64_t nz,int64_t ny,int64_t nx,int64_t src,double h,int64_t nc,double* residual) {
    try{auto g=new Graph(t,s,map,nz,ny,nx,src,h,nc);*residual=g->max_relative_residual;return g;}catch(...){return nullptr;}
}
API int adjoint_batch(void* handle,const int64_t* starts,int64_t count,double* rows,double* moments,int64_t* visits) {
    try{auto g=static_cast<Graph*>(handle);for(int64_t r=0;r<count;r++)g->row(starts[r],rows+r*g->nc,moments[r],visits[r]);return 0;}catch(...){return -1;}
}
API void adjoint_destroy(void* handle){delete static_cast<Graph*>(handle);}
