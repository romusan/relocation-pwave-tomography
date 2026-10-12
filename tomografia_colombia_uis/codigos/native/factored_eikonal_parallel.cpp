// Factored fast marching, Treister & Haber (2016), with discrete reverse AD.
// T=r*u; continuous source seeded at its surrounding cell vertices.
// Seed u is interpolated slowness at source; its derivative is included.
#include <algorithm>
#include <cmath>
#include <vector>
#include <queue>
#include <limits>
#include <cstring>
#include <exception>
#include <stdexcept>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif
struct Edge { int p[6]; double w[6], local; Edge():local(0){std::fill(p,p+6,-1);std::fill(w,w+6,0);} };
struct Term { double a,b,c1,c2; int p,q; };
struct Graph {
 int nz,ny,nx,n,order; double h; std::vector<double>s,r,u,t;
 std::vector<Edge> edges; std::vector<int> accepted,seeds; std::vector<double> sw;
 Graph(const double*slow,int z,int y,int x,double step,const double*src,int accuracy):nz(z),ny(y),nx(x),n(z*y*x),order(accuracy),h(step),s(slow,slow+n),r(n),u(n,1e100),t(n,1e100),edges(n){
  int dims[3]={nx,ny,nz},lo[3]; double f[3];
  for(int a=0;a<3;a++){if(src[a]<0||src[a]>dims[a]-1)throw std::runtime_error("source outside grid");lo[a]=std::min(int(std::floor(src[a])),dims[a]-2);f[a]=src[a]-lo[a];}
  double ss=0;
  for(int dz=0;dz<2;dz++)for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++){
   double w=(dx?f[0]:1-f[0])*(dy?f[1]:1-f[1])*(dz?f[2]:1-f[2]);
   if(w>1e-15){int v=((lo[2]+dz)*ny+lo[1]+dy)*nx+lo[0]+dx;seeds.push_back(v);sw.push_back(w);ss+=w*s[v];}
  }
  for(int k=0;k<nz;k++)for(int j=0;j<ny;j++)for(int i=0;i<nx;i++){int v=(k*ny+j)*nx+i;double dx=(i-src[0])*h,dy=(j-src[1])*h,dz=(k-src[2])*h;r[v]=std::sqrt(dx*dx+dy*dy+dz*dz);}
  std::vector<unsigned char> known(n,0);
  using Item=std::pair<double,int>;std::priority_queue<Item,std::vector<Item>,std::greater<Item>>heap;
  for(int v:seeds){known[v]=1;u[v]=ss;t[v]=r[v]*ss;accepted.push_back(v);}
  auto update=[&](int v){
   int c[3]={v%nx,(v/nx)%ny,v/(nx*ny)},stride[3]={1,nx,nx*ny};
   Term terms[3];int nt=0;
   for(int a=0;a<3;a++){
    int p=-1,sign=0;
    for(int sg:{-1,1})if(c[a]+sg>=0&&c[a]+sg<dims[a]){int w=v+sg*stride[a];if(known[w]&&(p<0||t[w]<t[p])){p=w;sign=sg;}}
    if(p<0)continue;
    int q=-1;if(order==2&&c[a]+2*sign>=0&&c[a]+2*sign<dims[a]){int w=v+2*sign*stride[a];if(known[w]&&t[w]<=t[p])q=w;}
    double grad=r[v]>0?(c[a]-src[a])*h/r[v]:0;
    double aa=(q<0?1.:1.5)*r[v]/h-sign*grad;
    if(aa<=1e-14)continue;
    double c1=(q<0?1.:2.)*r[v]/h,c2=q<0?0.:-.5*r[v]/h;
    terms[nt++]={aa,c1*u[p]+(q<0?0:c2*u[q]),c1,c2,p,q};
   }
   if(!nt)return;
   auto solve=[&](){
    std::sort(terms,terms+nt,[](Term a,Term b){return a.b/a.a<b.b/b.a;});
    double A=0,B=0,C=-s[v]*s[v],root=0;int count=0;
    for(int l=0;l<nt;l++){auto d=terms[l];A+=d.a*d.a;B+=d.a*d.b;C+=d.b*d.b;root=(B+std::sqrt(std::max(0.,B*B-A*C)))/A;count=l+1;if(l==nt-1||root<=terms[l+1].b/terms[l+1].a)break;}
    return std::pair<double,int>(root,count);
   };
   auto answer=solve();
   // Enforce monotone total T, reverting offending stencils to ordinary FM
   // in u coordinates (paper section 3.2). Keep derivatives of this branch.
   bool changed=false;
   for(int l=0;l<answer.second;l++){auto &d=terms[l];if(r[v]*answer.first+1e-12<t[d.p]){
     d.a=(d.q<0?1.:1.5)*r[v]/h;d.c1=(d.q<0?1.:2.)*r[d.p]/h;d.c2=d.q<0?0.:-.5*r[d.q]/h;d.b=d.c1*u[d.p]+(d.q<0?0:d.c2*u[d.q]);changed=true;
   }}
   if(changed)answer=solve();
   double tv=answer.first*r[v];
   if(tv>=t[v])return;
   double denom=0;for(int l=0;l<answer.second;l++)denom+=terms[l].a*(terms[l].a*answer.first-terms[l].b);
   if(denom<=0)return;
   Edge e;e.local=s[v]/denom;
   for(int l=0;l<answer.second;l++){auto d=terms[l];double g=(d.a*answer.first-d.b)/denom;e.p[2*l]=d.p;e.w[2*l]=d.c1*g;if(d.q>=0){e.p[2*l+1]=d.q;e.w[2*l+1]=d.c2*g;}}
   edges[v]=e;u[v]=answer.first;t[v]=tv;heap.push({tv,v});
  };
  auto neighbors=[&](int v){int c[3]={v%nx,(v/nx)%ny,v/(nx*ny)},str[3]={1,nx,nx*ny};for(int a=0;a<3;a++)for(int sg:{-1,1})if(c[a]+sg>=0&&c[a]+sg<dims[a]){int w=v+sg*str[a];if(!known[w])update(w);}};
  for(int v:seeds)neighbors(v);
  while(!heap.empty()){auto [time,v]=heap.top();heap.pop();if(known[v]||time!=t[v])continue;known[v]=1;accepted.push_back(v);neighbors(v);}
  if(int(accepted.size())!=n)throw std::runtime_error("unreached nodes");
 }
};
API void* factored_create(const double*s,int nz,int ny,int nx,double h,const double*src,int order,double*u,double*t){try{auto*g=new Graph(s,nz,ny,nx,h,src,order);std::copy(g->u.begin(),g->u.end(),u);std::copy(g->t.begin(),g->t.end(),t);return g;}catch(...){return nullptr;}}
API void factored_destroy(void*p){delete static_cast<Graph*>(p);}
API int factored_adjoint(void*p,const long long*ids,const double*weights,int nr,int nw,const long long*map,int nc,double*out){
 try{auto&g=*static_cast<Graph*>(p);
#pragma omp parallel for num_threads(4) schedule(dynamic)
for(int row=0;row<nr;row++){std::vector<double>lambda(g.n);
  std::fill(lambda.begin(),lambda.end(),0);double*dest=out+row*nc;std::fill(dest,dest+nc,0);
  for(int k=0;k<nw;k++)lambda[ids[row*nw+k]]+=weights[row*nw+k];
  for(auto it=g.accepted.rbegin();it!=g.accepted.rend();it++){int v=*it;double a=lambda[v];if(a==0)continue;auto&e=g.edges[v];dest[map[v]]+=a*e.local;for(int k=0;k<6;k++)if(e.p[k]>=0)lambda[e.p[k]]+=a*e.w[k];}
  double seedtotal=0;for(int v:g.seeds)seedtotal+=lambda[v];for(unsigned k=0;k<g.seeds.size();k++)dest[map[g.seeds[k]]]+=seedtotal*g.sw[k];
 }return 0;}catch(...){return 1;}
}
API int factored_projected(void*p,const long long*ids,const double*weights,int nr,int nw,const long long*indptr,const long long*indices,const double*values,int nc,double*out){
 try{auto&g=*static_cast<Graph*>(p);
#pragma omp parallel for num_threads(4) schedule(dynamic)
for(int row=0;row<nr;row++){std::vector<double>lambda(g.n),derivative(g.n);
  std::fill(lambda.begin(),lambda.end(),0);std::fill(derivative.begin(),derivative.end(),0);double*dest=out+row*nc;std::fill(dest,dest+nc,0);
  for(int k=0;k<nw;k++)lambda[ids[row*nw+k]]+=weights[row*nw+k];
  for(auto it=g.accepted.rbegin();it!=g.accepted.rend();it++){int v=*it;double a=lambda[v];if(a==0)continue;auto&e=g.edges[v];derivative[v]+=a*e.local;for(int k=0;k<6;k++)if(e.p[k]>=0)lambda[e.p[k]]+=a*e.w[k];}
  double seedtotal=0;for(int v:g.seeds)seedtotal+=lambda[v];for(unsigned k=0;k<g.seeds.size();k++)derivative[g.seeds[k]]+=seedtotal*g.sw[k];
  for(int v=0;v<g.n;v++)if(derivative[v]!=0)for(long long j=indptr[v];j<indptr[v+1];j++)dest[indices[j]]+=derivative[v]*values[j];
 }return 0;}catch(...){return 1;}
}
