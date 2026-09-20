// Eight-connected multi-source A*, unit lower bound Euclidean heuristic.
#include <vector>
#include <queue>
#include <cmath>
#include <limits>
struct Item {double f,d; int k; bool operator<(const Item&o)const{return f>o.f;}};
extern "C" int corridor_astar(const float* cost,int w,int h,double sx,double sy,double tx,double ty,double r,int* output,int* expanded){
 int count=w*h;std::vector<double>d(count,std::numeric_limits<double>::infinity());std::vector<int>parent(count,-1);std::priority_queue<Item>q;
 auto heuristic=[&](int x,int y){return std::max(0.,std::hypot(x-tx,y-ty)-r);};
 for(int y=std::max(0,(int)std::floor(sy-r));y<std::min(h,(int)std::ceil(sy+r)+1);y++)for(int x=std::max(0,(int)std::floor(sx-r));x<std::min(w,(int)std::ceil(sx+r)+1);x++)if(std::hypot(x-sx,y-sy)<=r){int k=y*w+x;d[k]=0;q.push({heuristic(x,y),0,k});}
 *expanded=0;
 while(!q.empty()){
 auto a=q.top();q.pop();if(a.d>d[a.k])continue;(*expanded)++;int x=a.k%w,y=a.k/w;
 if(std::hypot(x-tx,y-ty)<=r){int n=0;for(int k=a.k;k>=0;k=parent[k])output[n++]=k;return n;}
 for(int dy=-1;dy<=1;dy++)for(int dx=-1;dx<=1;dx++){if(!dx&&!dy)continue;int xx=x+dx,yy=y+dy;if(xx<0||xx>=w||yy<0||yy>=h)continue;int k=yy*w+xx;double nd=a.d+std::hypot(dx,dy)*(cost[a.k]+cost[k])*.5;if(nd<d[k]){d[k]=nd;parent[k]=a.k;q.push({nd+heuristic(xx,yy),nd,k});}}
 }return 0;
}
