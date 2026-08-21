// Radeon 构建失败二分诊断（v2，全量）。用法：cargo run --example diag_radeon
use ocl::{Device, Platform, ProQue};

const PRELUDE: &str = r#"
typedef long int64_t;
typedef unsigned long uint64_t;
"#;

fn try_build(name: &str, src: &str) -> bool {
    let platforms = Platform::list();
    let mut radeon: Option<Device> = None;
    for p in &platforms {
        let devs = Device::list_all(p).unwrap();
        for d in devs {
            let dn = d.name().unwrap().to_ascii_lowercase();
            if dn.contains("radeon") {
                radeon = Some(d);
                break;
            }
        }
        if radeon.is_some() {
            break;
        }
    }
    let dev = match radeon {
        Some(d) => d,
        None => {
            eprintln!("[{}] no Radeon device", name);
            return false;
        }
    };
    let full = format!("{}\n{}", PRELUDE, src);
    match ProQue::builder()
        .platform(Platform::default())
        .device(dev)
        .src(full.as_str())
        .build()
    {
        Ok(_) => {
            println!("[{}] BUILD OK", name);
            true
        }
        Err(e) => {
            println!("[{}] BUILD FAIL: {}", name, e);
            false
        }
    }
}

fn main() {
    let common = r#"
#define NL 8
constant uint P[8]={0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFEu,0xFFFFFC2Fu};
constant uint PM2[8]={0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFEu,0xFFFFFC2Du};
constant uint GX[8]={0x79BE667Eu,0xF9DCBBACu,0x55A06295u,0xCE870B07u,0x029BFCDBu,0x2DCE28D9u,0x59F2815Bu,0x16F81798u};
constant uint GY[8]={0x483ADA77u,0x26A3C465u,0x5DA4FBFCu,0x0E1108A8u,0xFD17B448u,0xA6855419u,0x9C47D08Fu,0xFB10D4B8u};
static inline int fe_ge(const uint* a,const uint* b){for(int i=0;i<NL;i++){if(a[i]>b[i])return 1;if(a[i]<b[i])return -1;}return 0;}
static inline int fe_is_zero(const uint* a){for(int i=0;i<NL;i++)if(a[i])return 0;return 1;}
static inline void fe_sub(uint* r,const uint* a,const uint* b){int64_t borrow=0;for(int i=NL-1;i>=0;i--){int64_t cur=(int64_t)a[i]-(int64_t)b[i]-borrow;if(cur<0){cur+=(1LL<<32);borrow=1;}else borrow=0;r[i]=(uint)cur;}}
static inline void fe_sub_mod(uint* r,const uint* a,const uint* b){uint d[8];int64_t borrow=0;for(int i=NL-1;i>=0;i--){int64_t cur=(int64_t)a[i]-(int64_t)b[i]-borrow;if(cur<0){cur+=(1LL<<32);borrow=1;}else borrow=0;d[i]=(uint)cur;}if(!borrow){for(int i=0;i<NL;i++)r[i]=d[i];}else{uint p[8];for(int i=0;i<NL;i++)p[i]=P[i];uint64_t c=0;for(int i=NL-1;i>=0;i--){c=(uint64_t)d[i]+(uint64_t)p[i]+(c>>32);r[i]=(uint)(c&0xFFFFFFFFULL);}}}
static inline void fe_add(uint* r,const uint* a,const uint* b){uint64_t c=0;for(int i=NL-1;i>=0;i--){c=(uint64_t)a[i]+(uint64_t)b[i]+(c>>32);r[i]=(uint)(c&0xFFFFFFFFULL);}uint p[8];for(int i=0;i<NL;i++)p[i]=P[i];if(c>>32){fe_sub(r,r,p);}else if(fe_ge(r,p)){fe_sub(r,r,p);}}
static void fe_reduce_512(uint* r,uint64_t* acc){
    for(int pass=0;pass<40;pass++){
        acc[14]+=acc[15]>>32;acc[15]&=0xFFFFFFFFULL;acc[13]+=acc[14]>>32;acc[14]&=0xFFFFFFFFULL;
        acc[12]+=acc[13]>>32;acc[13]&=0xFFFFFFFFULL;acc[11]+=acc[12]>>32;acc[12]&=0xFFFFFFFFULL;
        acc[10]+=acc[11]>>32;acc[11]&=0xFFFFFFFFULL;acc[9]+=acc[10]>>32;acc[10]&=0xFFFFFFFFULL;
        acc[8]+=acc[9]>>32;acc[9]&=0xFFFFFFFFULL;acc[7]+=acc[8]>>32;acc[8]&=0xFFFFFFFFULL;
        acc[6]+=acc[7]>>32;acc[7]&=0xFFFFFFFFULL;acc[5]+=acc[6]>>32;acc[6]&=0xFFFFFFFFULL;
        acc[4]+=acc[5]>>32;acc[5]&=0xFFFFFFFFULL;acc[3]+=acc[4]>>32;acc[4]&=0xFFFFFFFFULL;
        acc[2]+=acc[3]>>32;acc[3]&=0xFFFFFFFFULL;acc[1]+=acc[2]>>32;acc[2]&=0xFFFFFFFFULL;
        acc[0]+=acc[1]>>32;acc[1]&=0xFFFFFFFFULL;
        uint64_t hv0=acc[0],hv1=acc[1],hv2=acc[2],hv3=acc[3],hv4=acc[4],hv5=acc[5],hv6=acc[6],hv7=acc[7];
        uint64_t any=hv0|hv1|hv2|hv3|hv4|hv5|hv6|hv7;if(!any)break;
        acc[0]=0;acc[1]=0;acc[2]=0;acc[3]=0;acc[4]=0;acc[5]=0;acc[6]=0;acc[7]=0;
        acc[7]+=hv0;acc[8]+=hv0*977ULL;acc[8]+=hv1;acc[9]+=hv1*977ULL;acc[9]+=hv2;acc[10]+=hv2*977ULL;
        acc[10]+=hv3;acc[11]+=hv3*977ULL;acc[11]+=hv4;acc[12]+=hv4*977ULL;acc[12]+=hv5;acc[13]+=hv5*977ULL;
        acc[13]+=hv6;acc[14]+=hv6*977ULL;acc[14]+=hv7;acc[15]+=hv7*977ULL;
    }
    uint rr[8];for(int i=0;i<8;i++)rr[i]=(uint)acc[8+i];uint p[8];for(int i=0;i<NL;i++)p[i]=P[i];
    while(fe_ge(rr,p))fe_sub(rr,rr,p);for(int i=0;i<NL;i++)r[i]=rr[i];
}
static void fe_mul(uint* r,const uint* a,const uint* b){
    uint64_t acc[16];for(int i=0;i<16;i++)acc[i]=0;
    acc[0]+=((uint64_t)a[0]*(uint64_t)b[0])>>32;acc[1]+=((uint64_t)a[0]*(uint64_t)b[0])&0xFFFFFFFFULL;
    acc[14]+=((uint64_t)a[7]*(uint64_t)b[7])>>32;acc[15]+=((uint64_t)a[7]*(uint64_t)b[7])&0xFFFFFFFFULL;
    fe_reduce_512(r,acc);
}
static void fe_sqr(uint* r,const uint* a){fe_mul(r,a,a);}
static void fe_pow(uint* r,const uint* a,const uint* e){
    uint res[8]={0,0,0,0,0,0,0,1};uint base[8];for(int i=0;i<NL;i++)base[i]=a[i];
    for(int i=0;i<NL;i++)for(int bit=31;bit>=0;bit--){
        uint t[8];fe_mul(t,res,res);for(int k=0;k<NL;k++)res[k]=t[k];
        if((e[i]>>bit)&1u){uint u[8];fe_mul(u,res,base);for(int k=0;k<NL;k++)res[k]=u[k];}
    }
    for(int i=0;i<NL;i++)r[i]=res[i];
}
static void fe_inv(uint* r,const uint* a){uint pm2[8];for(int i=0;i<NL;i++)pm2[i]=PM2[i];fe_pow(r,a,pm2);}
static void jdouble(uint* RX,uint* RY,uint* RZ,const uint* PX,const uint* PY,const uint* PZ){
    if(fe_is_zero(PZ)||fe_is_zero(PY)){uint one[8]={0,0,0,0,0,0,0,1};for(int i=0;i<NL;i++){RX[i]=one[i];RY[i]=one[i];}RZ[7]=0;return;}
    uint delta[8],gamma[8],beta[8],alpha[8],t1[8],t2[8],t3[8];
    fe_sqr(delta,PZ);fe_sqr(gamma,PY);fe_mul(beta,PX,gamma);
    fe_sqr(t1,PX);fe_add(alpha,t1,t1);fe_add(alpha,alpha,t1);
    fe_sqr(t1,alpha);fe_add(t2,beta,beta);fe_add(t2,t2,t2);fe_add(t2,t2,t2);
    fe_sub_mod(RX,t1,t2);fe_mul(t1,PY,PZ);fe_add(RZ,t1,t1);
    fe_add(t1,beta,beta);fe_add(t1,t1,t1);fe_sub_mod(t2,t1,RX);fe_mul(t3,alpha,t2);
    fe_sqr(t1,gamma);fe_add(t2,t1,t1);fe_add(t2,t2,t2);fe_add(t2,t2,t2);fe_sub_mod(RY,t3,t2);
}
static void jadd_mixed(uint* RX,uint* RY,uint* RZ,const uint* PX,const uint* PY,const uint* PZ,constant uint* QX,constant uint* QY){
    uint qx[8],qy[8];for(int i=0;i<NL;i++){qx[i]=QX[i];qy[i]=QY[i];}
    if(fe_is_zero(PZ)){for(int i=0;i<NL;i++){RX[i]=qx[i];RY[i]=qy[i];}uint one[8]={0,0,0,0,0,0,0,1};for(int i=0;i<NL;i++)RZ[i]=one[i];return;}
    uint z2[8],z3[8],u2[8],s2[8],h[8],rr[8];fe_sqr(z2,PZ);fe_mul(z3,z2,PZ);fe_mul(u2,qx,z2);fe_mul(s2,qy,z3);
    fe_sub_mod(h,u2,PX);fe_sub_mod(rr,s2,PY);
    if(fe_is_zero(h)){if(fe_is_zero(rr)){jdouble(RX,RY,RZ,PX,PY,PZ);}else{uint one[8]={0,0,0,0,0,0,0,1};for(int i=0;i<NL;i++){RX[i]=one[i];RY[i]=one[i];}RZ[7]=0;}return;}
    uint h2[8],h3[8],u1h2[8],t1[8],t2[8];fe_sqr(h2,h);fe_mul(h3,h2,h);fe_mul(u1h2,PX,h2);
    fe_sqr(t1,rr);fe_sub_mod(t2,t1,h3);fe_add(t1,u1h2,u1h2);fe_sub_mod(RX,t2,t1);
    fe_sub_mod(t1,u1h2,RX);fe_mul(t2,rr,t1);fe_mul(t1,PY,h3);fe_sub_mod(RY,t2,t1);fe_mul(RZ,PZ,h);
}
static void jto_affine(uint* Qx,uint* Qy,const uint* X,const uint* Y,const uint* Z){
    if(fe_is_zero(Z)){for(int i=0;i<NL;i++){Qx[i]=0;Qy[i]=0;}return;}
    uint z2[8],z3[8],invz2[8],invz3[8];fe_sqr(z2,Z);fe_mul(z3,z2,Z);fe_inv(invz2,z2);fe_inv(invz3,z3);
    fe_mul(Qx,X,invz2);fe_mul(Qy,Y,invz3);
}
static void scalar_mul(uint* Qx,uint* Qy,const uint* k){
    uint RX[8],RY[8],RZ[8];uint one[8]={0,0,0,0,0,0,0,1};
    for(int i=0;i<NL;i++){RX[i]=one[i];RY[i]=one[i];}RZ[7]=0;
    int Rinf=1;
    for(int i=0;i<NL;i++)for(int bit=31;bit>=0;bit--){
        uint kb=(k[i]>>bit)&1u;
        if(!Rinf){uint TX[8],TY[8],TZ[8];jdouble(TX,TY,TZ,RX,RY,RZ);for(int t=0;t<NL;t++){RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}
        if(kb){if(Rinf){for(int t=0;t<NL;t++){RX[t]=GX[t];RY[t]=GY[t];}for(int t=0;t<NL;t++)RZ[t]=one[t];Rinf=0;}
            else{uint TX[8],TY[8],TZ[8];jadd_mixed(TX,TY,TZ,RX,RY,RZ,GX,GY);for(int t=0;t<NL;t++){RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}
    }
    jto_affine(Qx,Qy,RX,RY,RZ);
}
"#;

    let _ = try_build("A_common_only", &format!("{}\n__kernel void k(__global uint* o){{uint a[8]={{0,0,0,0,0,0,0,5}};uint r[8];fe_inv(r,a);for(int i=0;i<8;i++)o[i]=r[i];}}\n", common));

    let _ = try_build("B_plus_scalar_mul", &format!("{}\n__kernel void k(__global uint* o){{uint k[8]={{0,0,0,0,0,0,0,2}};uint Qx[8],Qy[8];scalar_mul(Qx,Qy,k);for(int i=0;i<8;i++)o[i]=Qx[i];}}\n", common));

    let _ = try_build("C_plus_keccak", &format!("{}\nstatic inline uint64_t rotl64(uint64_t x,uint64_t n){{return (x<<n)|(x>>(64-n));}}\nstatic void keccak_f(uint64_t* st){{const uint64_t RC[24]={{0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808aULL,0x8000000080008000ULL,0x000000000000808bULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,0x000000000000008aULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000aULL,0x000000008000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800aULL,0x800000008000000aULL,0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL}};const int ROTC[24]={{1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44}};const int PILN[24]={{10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1}};volatile int rounds=24;for(int round=0;round<rounds;round++){{uint64_t bc[5];for(int i=0;i<5;i++)bc[i]=st[i]^st[i+5]^st[i+10]^st[i+15]^st[i+20];for(int i=0;i<5;i++){{uint64_t t=bc[(i+4)%5]^rotl64(bc[(i+1)%5],(uint64_t)1);for(int j=0;j<5;j++)st[i+5*j]^=t;}}uint64_t tmp=st[1];for(int i=0;i<24;i++){{int j=PILN[i];uint64_t t=st[j];st[j]=rotl64(tmp,(uint64_t)ROTC[i]);tmp=t;}}for(int j=0;j<5;j++){{uint64_t tc[5];for(int i=0;i<5;i++)tc[i]=st[j*5+i];for(int i=0;i<5;i++)st[j*5+i]=tc[i]^((~tc[(i+1)%5])&tc[(i+2)%5]);}}st[0]^=RC[round];}}}}\nstatic void keccak256_addr(const uint* x,const uint* y,__global uchar* out_addr){{uint64_t st[25];for(int i=0;i<25;i++)st[i]=0;for(int lane=0;lane<8;lane++){{uint a=(lane<4)?x[2*lane]:y[2*(lane-4)];uint b=(lane<4)?x[2*lane+1]:y[2*(lane-4)+1];uint64_t v=0;v|=((uint64_t)((a>>24)&0xFF));v|=((uint64_t)((a>>16)&0xFF))<<8;v|=((uint64_t)((a>>8)&0xFF))<<16;v|=((uint64_t)(a&0xFF))<<24;v|=((uint64_t)((b>>24)&0xFF))<<32;v|=((uint64_t)((b>>16)&0xFF))<<40;v|=((uint64_t)((b>>8)&0xFF))<<48;v|=((uint64_t)(b&0xFF))<<56;st[lane]=v;}}st[8]^=(uint64_t)0x01;st[16]^=(uint64_t)0x8000000000000000ULL;keccak_f(st);for(int i=0;i<20;i++)out_addr[i]=(uchar)(st[(12+i)/8]>>(8*((12+i)%8)));}}\n__kernel void k(__global uint* o,__global uchar* ad){{uint k[8]={{0,0,0,0,0,0,0,2}};uint Qx[8],Qy[8];scalar_mul(Qx,Qy,k);keccak256_addr(Qx,Qy,ad);for(int i=0;i<8;i++)o[i]=Qx[i];}}\n", common));

    // D: scalar_mul 但只调 jdouble（去掉 jadd_mixed 与 jto_affine 调用）
    let scalar_d = r#"
static void scalar_mul_d(uint* Qx,uint* Qy,const uint* k){
    uint RX[8],RY[8],RZ[8];uint one[8]={0,0,0,0,0,0,0,1};
    for(int i=0;i<NL;i++){RX[i]=one[i];RY[i]=one[i];}RZ[7]=0;
    int Rinf=1;
    for(int i=0;i<NL;i++)for(int bit=31;bit>=0;bit--){
        uint kb=(k[i]>>bit)&1u;
        if(!Rinf){uint TX[8],TY[8],TZ[8];jdouble(TX,TY,TZ,RX,RY,RZ);for(int t=0;t<NL;t++){RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}
        if(kb){if(Rinf){for(int t=0;t<NL;t++){RX[t]=GX[t];RY[t]=GY[t];}for(int t=0;t<NL;t++)RZ[t]=one[t];Rinf=0;}}
    }
    for(int i=0;i<NL;i++){Qx[i]=RX[i];Qy[i]=RY[i];}
}
"#;
    let _ = try_build("D_jdouble_only", &format!("{}{}\n__kernel void k(__global uint* o){{uint k[8]={{0,0,0,0,0,0,0,2}};uint Qx[8],Qy[8];scalar_mul_d(Qx,Qy,k);for(int i=0;i<8;i++)o[i]=Qx[i];}}\n", common, scalar_d));

    // E: D + jadd_mixed（去掉 jto_affine/fe_inv）
    let _ = try_build("E_jadd_only", &format!("{}\n__kernel void k(__global uint* o){{uint k[8]={{0,0,0,0,0,0,0,2}};uint Qx[8],Qy[8];uint RX[8],RY[8],RZ[8];uint one[8]={{0,0,0,0,0,0,0,1}};for(int i=0;i<NL;i++){{RX[i]=one[i];RY[i]=one[i];}}RZ[7]=0;int Rinf=1;for(int i=0;i<NL;i++)for(int bit=31;bit>=0;bit--){{uint kb=(k[i]>>bit)&1u;if(!Rinf){{uint TX[8],TY[8],TZ[8];jdouble(TX,TY,TZ,RX,RY,RZ);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}if(kb){{if(Rinf){{for(int t=0;t<NL;t++){{RX[t]=GX[t];RY[t]=GY[t];}}for(int t=0;t<NL;t++)RZ[t]=one[t];Rinf=0;}}else{{uint TX[8],TY[8],TZ[8];jadd_mixed(TX,TY,TZ,RX,RY,RZ,GX,GY);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}}}for(int i=0;i<NL;i++){{Qx[i]=RX[i];Qy[i]=RY[i];}}}}\n", common));

    // F: jadd_mixed 去掉递归 jdouble 调用（P==Q 时直接返回 infinity），其余不变
    let jadd_f = r#"
static void jadd_mixed_f(uint* RX,uint* RY,uint* RZ,const uint* PX,const uint* PY,const uint* PZ,constant uint* QX,constant uint* QY){
    uint qx[8],qy[8];for(int i=0;i<NL;i++){qx[i]=QX[i];qy[i]=QY[i];}
    if(fe_is_zero(PZ)){for(int i=0;i<NL;i++){RX[i]=qx[i];RY[i]=qy[i];}uint one[8]={0,0,0,0,0,0,0,1};for(int i=0;i<NL;i++)RZ[i]=one[i];return;}
    uint z2[8],z3[8],u2[8],s2[8],h[8],rr[8];fe_sqr(z2,PZ);fe_mul(z3,z2,PZ);fe_mul(u2,qx,z2);fe_mul(s2,qy,z3);
    fe_sub_mod(h,u2,PX);fe_sub_mod(rr,s2,PY);
    if(fe_is_zero(h)){if(fe_is_zero(rr)){uint one[8]={0,0,0,0,0,0,0,1};for(int i=0;i<NL;i++){RX[i]=one[i];RY[i]=one[i];}RZ[7]=0;return;}else{uint one[8]={0,0,0,0,0,0,0,1};for(int i=0;i<NL;i++){RX[i]=one[i];RY[i]=one[i];}RZ[7]=0;return;}}
    uint h2[8],h3[8],u1h2[8],t1[8],t2[8];fe_sqr(h2,h);fe_mul(h3,h2,h);fe_mul(u1h2,PX,h2);
    fe_sqr(t1,rr);fe_sub_mod(t2,t1,h3);fe_add(t1,u1h2,u1h2);fe_sub_mod(RX,t2,t1);
    fe_sub_mod(t1,u1h2,RX);fe_mul(t2,rr,t1);fe_mul(t1,PY,h3);fe_sub_mod(RY,t2,t1);fe_mul(RZ,PZ,h);
}
"#;
    let _ = try_build("F_jadd_no_recurse", &format!("{}{}\n__kernel void k(__global uint* o){{uint k[8]={{0,0,0,0,0,0,0,2}};uint Qx[8],Qy[8];uint RX[8],RY[8],RZ[8];uint one[8]={{0,0,0,0,0,0,0,1}};for(int i=0;i<NL;i++){{RX[i]=one[i];RY[i]=one[i];}}RZ[7]=0;int Rinf=1;for(int i=0;i<NL;i++)for(int bit=31;bit>=0;bit--){{uint kb=(k[i]>>bit)&1u;if(!Rinf){{uint TX[8],TY[8],TZ[8];jdouble(TX,TY,TZ,RX,RY,RZ);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}if(kb){{if(Rinf){{for(int t=0;t<NL;t++){{RX[t]=GX[t];RY[t]=GY[t];}}for(int t=0;t<NL;t++)RZ[t]=one[t];Rinf=0;}}else{{uint TX[8],TY[8],TZ[8];jadd_mixed_f(TX,TY,TZ,RX,RY,RZ,GX,GY);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}}}for(int i=0;i<NL;i++){{Qx[i]=RX[i];Qy[i]=RY[i];}}}}\n", common, jadd_f));

    // G: jadd_mixed 极简版（只从 constant 复制 + 直接返回，不调任何 fe_*）
    let jadd_g = r#"
static void jadd_mixed_g(uint* RX,uint* RY,uint* RZ,const uint* PX,const uint* PY,const uint* PZ,constant uint* QX,constant uint* QY){
    uint qx[8],qy[8];for(int i=0;i<NL;i++){qx[i]=QX[i];qy[i]=QY[i];}
    for(int i=0;i<NL;i++){RX[i]=qx[i];RY[i]=qy[i];}uint one[8]={0,0,0,0,0,0,0,1};for(int i=0;i<NL;i++)RZ[i]=one[i];
}
"#;
    let _ = try_build("G_jadd_trivial", &format!("{}{}\n__kernel void k(__global uint* o){{uint k[8]={{0,0,0,0,0,0,0,2}};uint Qx[8],Qy[8];uint RX[8],RY[8],RZ[8];uint one[8]={{0,0,0,0,0,0,0,1}};for(int i=0;i<NL;i++){{RX[i]=one[i];RY[i]=one[i];}}RZ[7]=0;int Rinf=1;for(int i=0;i<NL;i++)for(int bit=31;bit>=0;bit--){{uint kb=(k[i]>>bit)&1u;if(!Rinf){{uint TX[8],TY[8],TZ[8];jdouble(TX,TY,TZ,RX,RY,RZ);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}if(kb){{if(Rinf){{for(int t=0;t<NL;t++){{RX[t]=GX[t];RY[t]=GY[t];}}for(int t=0;t<NL;t++)RZ[t]=one[t];Rinf=0;}}else{{uint TX[8],TY[8],TZ[8];jadd_mixed_g(TX,TY,TZ,RX,RY,RZ,GX,GY);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}}}for(int i=0;i<NL;i++){{Qx[i]=RX[i];Qy[i]=RY[i];}}}}\n", common, jadd_g));

    // H: jadd_mixed 参数改为非 constant (const uint*)，调用处先把 GX/GY 复制到 private 再传
    let jadd_h = r#"
static void jadd_mixed_h(uint* RX,uint* RY,uint* RZ,const uint* PX,const uint* PY,const uint* PZ,const uint* QX,const uint* QY){
    uint qx[8],qy[8];for(int i=0;i<NL;i++){qx[i]=QX[i];qy[i]=QY[i];}
    if(fe_is_zero(PZ)){for(int i=0;i<NL;i++){RX[i]=qx[i];RY[i]=qy[i];}uint one[8]={0,0,0,0,0,0,0,1};for(int i=0;i<NL;i++)RZ[i]=one[i];return;}
    uint z2[8],z3[8],u2[8],s2[8],h[8],rr[8];fe_sqr(z2,PZ);fe_mul(z3,z2,PZ);fe_mul(u2,qx,z2);fe_mul(s2,qy,z3);
    fe_sub_mod(h,u2,PX);fe_sub_mod(rr,s2,PY);
    if(fe_is_zero(h)){if(fe_is_zero(rr)){jdouble(RX,RY,RZ,PX,PY,PZ);}else{uint one[8]={0,0,0,0,0,0,0,1};for(int i=0;i<NL;i++){RX[i]=one[i];RY[i]=one[i];}RZ[7]=0;}return;}
    uint h2[8],h3[8],u1h2[8],t1[8],t2[8];fe_sqr(h2,h);fe_mul(h3,h2,h);fe_mul(u1h2,PX,h2);
    fe_sqr(t1,rr);fe_sub_mod(t2,t1,h3);fe_add(t1,u1h2,u1h2);fe_sub_mod(RX,t2,t1);
    fe_sub_mod(t1,u1h2,RX);fe_mul(t2,rr,t1);fe_mul(t1,PY,h3);fe_sub_mod(RY,t2,t1);fe_mul(RZ,PZ,h);
}
"#;
    let _ = try_build("H_jadd_private_arg", &format!("{}{}\n__kernel void k(__global uint* o){{uint k[8]={{0,0,0,0,0,0,0,2}};uint Qx[8],Qy[8];uint RX[8],RY[8],RZ[8];uint one[8]={{0,0,0,0,0,0,0,1}};for(int i=0;i<NL;i++){{RX[i]=one[i];RY[i]=one[i];}}RZ[7]=0;uint gxp[8],gyp[8];for(int i=0;i<NL;i++){{gxp[i]=GX[i];gyp[i]=GY[i];}}int Rinf=1;for(int i=0;i<NL;i++)for(int bit=31;bit>=0;bit--){{uint kb=(k[i]>>bit)&1u;if(!Rinf){{uint TX[8],TY[8],TZ[8];jdouble(TX,TY,TZ,RX,RY,RZ);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}if(kb){{if(Rinf){{for(int t=0;t<NL;t++){{RX[t]=gxp[t];RY[t]=gyp[t];}}for(int t=0;t<NL;t++)RZ[t]=one[t];Rinf=0;}}else{{uint TX[8],TY[8],TZ[8];jadd_mixed_h(TX,TY,TZ,RX,RY,RZ,gxp,gyp);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}}}for(int i=0;i<NL;i++){{Qx[i]=RX[i];Qy[i]=RY[i];}}}}\n", common, jadd_h));

    // I: scalar_mul 但循环只跑 i=0（32 次迭代而非 256 次），调原版 jadd_mixed (constant 参数)
    let _ = try_build("I_scalar_32iter", &format!("{}\n__kernel void k(__global uint* o){{uint k[8]={{0,0,0,0,0,0,0,2}};uint Qx[8],Qy[8];uint RX[8],RY[8],RZ[8];uint one[8]={{0,0,0,0,0,0,0,1}};for(int i=0;i<1;i++)for(int bit=31;bit>=0;bit--){{uint kb=(k[i]>>bit)&1u;if(!Rinf_placeholder){{}}if(kb){{if(Rinf_ph2){{for(int t=0;t<NL;t++){{RX[t]=GX[t];RY[t]=GY[t];}}for(int t=0;t<NL;t++)RZ[t]=one[t];}}else{{uint TX[8],TY[8],TZ[8];jadd_mixed(TX,TY,TZ,RX,RY,RZ,GX,GY);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}}}for(int i=0;i<NL;i++){{Qx[i]=RX[i];Qy[i]=RY[i];}}}}\n", common));

    // J: 同 I 但循环 256 次 (i<NL) 且调 jadd_mixed，验证循环深度
    let _ = try_build("J_scalar_256iter", &format!("{}\n__kernel void k(__global uint* o){{uint k[8]={{0,0,0,0,0,0,0,2}};uint Qx[8],Qy[8];uint RX[8],RY[8],RZ[8];uint one[8]={{0,0,0,0,0,0,0,1}};int Rinf=1;for(int i=0;i<NL;i++)for(int bit=31;bit>=0;bit--){{uint kb=(k[i]>>bit)&1u;if(!Rinf){{uint TX[8],TY[8],TZ[8];jdouble(TX,TY,TZ,RX,RY,RZ);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}if(kb){{if(Rinf){{for(int t=0;t<NL;t++){{RX[t]=GX[t];RY[t]=GY[t];}}for(int t=0;t<NL;t++)RZ[t]=one[t];Rinf=0;}}else{{uint TX[8],TY[8],TZ[8];jadd_mixed(TX,TY,TZ,RX,RY,RZ,GX,GY);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}}}for(int i=0;i<NL;i++){{Qx[i]=RX[i];Qy[i]=RY[i];}}}}\n", common));

    // K: jadd_mixed 全功能，参数全 const uint*（非 constant），函数内直接用 QX/QY（不复制到 qx/qy），
    //    调用处复制 GX/GY 到 private 后传入。测是否去掉 constant 复制循环即可。
    let jadd_k = r#"
static void jadd_mixed_k(uint* RX,uint* RY,uint* RZ,const uint* PX,const uint* PY,const uint* PZ,const uint* QX,const uint* QY){
    if(fe_is_zero(PZ)){for(int i=0;i<NL;i++){RX[i]=QX[i];RY[i]=QY[i];}uint one[8]={0,0,0,0,0,0,0,1};for(int i=0;i<NL;i++)RZ[i]=one[i];return;}
    uint z2[8],z3[8],u2[8],s2[8],h[8],rr[8];fe_sqr(z2,PZ);fe_mul(z3,z2,PZ);fe_mul(u2,QX,z2);fe_mul(s2,QY,z3);
    fe_sub_mod(h,u2,PX);fe_sub_mod(rr,s2,PY);
    if(fe_is_zero(h)){if(fe_is_zero(rr)){jdouble(RX,RY,RZ,PX,PY,PZ);}else{uint one[8]={0,0,0,0,0,0,0,1};for(int i=0;i<NL;i++){RX[i]=one[i];RY[i]=one[i];}RZ[7]=0;}return;}
    uint h2[8],h3[8],u1h2[8],t1[8],t2[8];fe_sqr(h2,h);fe_mul(h3,h2,h);fe_mul(u1h2,PX,h2);
    fe_sqr(t1,rr);fe_sub_mod(t2,t1,h3);fe_add(t1,u1h2,u1h2);fe_sub_mod(RX,t2,t1);
    fe_sub_mod(t1,u1h2,RX);fe_mul(t2,rr,t1);fe_mul(t1,PY,h3);fe_sub_mod(RY,t2,t1);fe_mul(RZ,PZ,h);
}
"#;
    let _ = try_build("K_jadd_nocopy", &format!("{}{}\n__kernel void k(__global uint* o){{uint k[8]={{0,0,0,0,0,0,0,2}};uint Qx[8],Qy[8];uint RX[8],RY[8],RZ[8];uint one[8]={{0,0,0,0,0,0,0,1}};for(int i=0;i<NL;i++){{RX[i]=one[i];RY[i]=one[i];}}RZ[7]=0;uint gxp[8],gyp[8];for(int i=0;i<NL;i++){{gxp[i]=GX[i];gyp[i]=GY[i];}}int Rinf=1;for(int i=0;i<NL;i++)for(int bit=31;bit>=0;bit--){{uint kb=(k[i]>>bit)&1u;if(!Rinf){{uint TX[8],TY[8],TZ[8];jdouble(TX,TY,TZ,RX,RY,RZ);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}if(kb){{if(Rinf){{for(int t=0;t<NL;t++){{RX[t]=gxp[t];RY[t]=gyp[t];}}for(int t=0;t<NL;t++)RZ[t]=one[t];Rinf=0;}}else{{uint TX[8],TY[8],TZ[8];jadd_mixed_k(TX,TY,TZ,RX,RY,RZ,gxp,gyp);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}}}for(int i=0;i<NL;i++){{Qx[i]=RX[i];Qy[i]=RY[i];}}}}\n", common, jadd_k));

    // L: 重量级 jdouble（连续调 jdouble 两次，增加局部数组/代码量），测是否代码量触发
    let heavy = r#"
static void jdouble2(uint* RX,uint* RY,uint* RZ,const uint* PX,const uint* PY,const uint* PZ){
    uint TX[8],TY[8],TZ[8];jdouble(TX,TY,TZ,PX,PY,PZ);
    jdouble(RX,RY,RZ,TX,TY,TZ);
}
"#;
    let _ = try_build("L_heavy_jdouble", &format!("{}{}\n__kernel void k(__global uint* o){{uint k[8]={{0,0,0,0,0,0,0,2}};uint Qx[8],Qy[8];uint RX[8],RY[8],RZ[8];uint one[8]={{0,0,0,0,0,0,0,1}};for(int i=0;i<NL;i++){{RX[i]=one[i];RY[i]=one[i];}}RZ[7]=0;int Rinf=1;for(int i=0;i<NL;i++)for(int bit=31;bit>=0;bit--){{uint kb=(k[i]>>bit)&1u;if(!Rinf){{uint TX[8],TY[8],TZ[8];jdouble2(TX,TY,TZ,RX,RY,RZ);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}if(kb){{if(Rinf){{for(int t=0;t<NL;t++){{RX[t]=GX[t];RY[t]=GY[t];}}for(int t=0;t<NL;t++)RZ[t]=one[t];Rinf=0;}}else{{uint TX[8],TY[8],TZ[8];jdouble2(TX,TY,TZ,RX,RY,RZ);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}}}for(int i=0;i<NL;i++){{Qx[i]=RX[i];Qy[i]=RY[i];}}}}\n", common, heavy));

    // M: jadd_mixed 但去掉所有 fe_mul，只保留 fe_sqr + fe_sub_mod + fe_add（测 fe_mul 是否触发）
    let jadd_m = r#"
static void jadd_mixed_m(uint* RX,uint* RY,uint* RZ,const uint* PX,const uint* PY,const uint* PZ,const uint* QX,const uint* QY){
    uint qx[8],qy[8];for(int i=0;i<NL;i++){qx[i]=QX[i];qy[i]=QY[i];}
    if(fe_is_zero(PZ)){for(int i=0;i<NL;i++){RX[i]=qx[i];RY[i]=qy[i];}uint one[8]={0,0,0,0,0,0,0,1};for(int i=0;i<NL;i++)RZ[i]=one[i];return;}
    uint z2[8],u2[8],h[8],rr[8];fe_sqr(z2,PZ);fe_sqr(u2,qx);fe_sqr(rr,qy);
    fe_sub_mod(h,u2,PX);fe_sub_mod(rr,rr,PY);
    if(fe_is_zero(h)){uint one[8]={0,0,0,0,0,0,0,1};for(int i=0;i<NL;i++){RX[i]=one[i];RY[i]=one[i];}RZ[7]=0;return;}
    uint h2[8],u1h2[8],t1[8],t2[8];fe_sqr(h2,h);fe_sqr(u1h2,PX);
    fe_add(t1,u1h2,u1h2);fe_sub_mod(RX,h2,t1);
    fe_sub_mod(t1,u1h2,RX);fe_add(t2,rr,t1);fe_sub_mod(RY,t2,t1);fe_add(RZ,PZ,h);
}
"#;
    let _ = try_build("M_jadd_no_femul", &format!("{}{}\n__kernel void k(__global uint* o){{uint k[8]={{0,0,0,0,0,0,0,2}};uint Qx[8],Qy[8];uint RX[8],RY[8],RZ[8];uint one[8]={{0,0,0,0,0,0,0,1}};for(int i=0;i<NL;i++){{RX[i]=one[i];RY[i]=one[i];}}RZ[7]=0;uint gxp[8],gyp[8];for(int i=0;i<NL;i++){{gxp[i]=GX[i];gyp[i]=GY[i];}}int Rinf=1;for(int i=0;i<NL;i++)for(int bit=31;bit>=0;bit--){{uint kb=(k[i]>>bit)&1u;if(!Rinf){{uint TX[8],TY[8],TZ[8];jdouble(TX,TY,TZ,RX,RY,RZ);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}if(kb){{if(Rinf){{for(int t=0;t<NL;t++){{RX[t]=gxp[t];RY[t]=gyp[t];}}for(int t=0;t<NL;t++)RZ[t]=one[t];Rinf=0;}}else{{uint TX[8],TY[8],TZ[8];jadd_mixed_m(TX,TY,TZ,RX,RY,RZ,gxp,gyp);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}}}for(int i=0;i<NL;i++){{Qx[i]=RX[i];Qy[i]=RY[i];}}}}\n", common, jadd_m));

    // N: scalar_mul 循环上界改为运行时参数 nlimbs（非编译期常量），key 从 __global 读，测循环展开假设
    let scalar_n = r#"
static void scalar_mul_n(uint* Qx,uint* Qy,__global const uint* k,int nlimbs){
    uint RX[8],RY[8],RZ[8];uint one[8]={0,0,0,0,0,0,0,1};
    for(int i=0;i<NL;i++){RX[i]=one[i];RY[i]=one[i];}RZ[7]=0;
    int Rinf=1;
    for(int i=0;i<nlimbs;i++)for(int bit=31;bit>=0;bit--){
        uint kb=(k[i]>>bit)&1u;
        if(!Rinf){uint TX[8],TY[8],TZ[8];jdouble(TX,TY,TZ,RX,RY,RZ);for(int t=0;t<NL;t++){RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}
        if(kb){if(Rinf){for(int t=0;t<NL;t++){RX[t]=GX[t];RY[t]=GY[t];}for(int t=0;t<NL;t++)RZ[t]=one[t];Rinf=0;}
            else{uint TX[8],TY[8],TZ[8];jadd_mixed(TX,TY,TZ,RX,RY,RZ,GX,GY);for(int t=0;t<NL;t++){RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}
    }
    jto_affine(Qx,Qy,RX,RY,RZ);
}
"#;
    let _ = try_build("N_runtime_loop", &format!("{}{}\n__kernel void k(__global uint* o,__global uint* kin){{uint Qx[8],Qy[8];scalar_mul_n(Qx,Qy,kin,8);for(int i=0;i<8;i++)o[i]=Qx[i];}}\n", common, scalar_n));

    // P: jadd 风格的 fe_mul 链式（z2=sqr(PZ); z3=mul(z2,PZ); u2=mul(qx,z2); s2=mul(qy,z3)），然后返回，测是否 fe_mul 链式触发
    let jadd_p = r#"
static void jadd_mixed_p(uint* RX,uint* RY,uint* RZ,const uint* PX,const uint* PY,const uint* PZ,const uint* QX,const uint* QY){
    uint qx[8],qy[8];for(int i=0;i<NL;i++){qx[i]=QX[i];qy[i]=QY[i];}
    uint z2[8],z3[8],u2[8],s2[8];fe_sqr(z2,PZ);fe_mul(z3,z2,PZ);fe_mul(u2,qx,z2);fe_mul(s2,qy,z3);
    for(int i=0;i<NL;i++){RX[i]=u2[i];RY[i]=s2[i];RZ[i]=z3[i];}
}
"#;
    let _ = try_build("P_jadd_femul_chain", &format!("{}{}\n__kernel void k(__global uint* o){{uint k[8]={{0,0,0,0,0,0,0,2}};uint Qx[8],Qy[8];uint RX[8],RY[8],RZ[8];uint one[8]={{0,0,0,0,0,0,0,1}};for(int i=0;i<NL;i++){{RX[i]=one[i];RY[i]=one[i];}}RZ[7]=0;uint gxp[8],gyp[8];for(int i=0;i<NL;i++){{gxp[i]=GX[i];gyp[i]=GY[i];}}int Rinf=1;for(int i=0;i<NL;i++)for(int bit=31;bit>=0;bit--){{uint kb=(k[i]>>bit)&1u;if(!Rinf){{uint TX[8],TY[8],TZ[8];jdouble(TX,TY,TZ,RX,RY,RZ);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}if(kb){{if(Rinf){{for(int t=0;t<NL;t++){{RX[t]=gxp[t];RY[t]=gyp[t];}}for(int t=0;t<NL;t++)RZ[t]=one[t];Rinf=0;}}else{{uint TX[8],TY[8],TZ[8];jadd_mixed_p(TX,TY,TZ,RX,RY,RZ,gxp,gyp);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}}}for(int i=0;i<NL;i++){{Qx[i]=RX[i];Qy[i]=RY[i];}}}}\n", common, jadd_p));

    // Q: jdouble 但手动内联展开两次（确认 L 的结论：代码量/内联深度触发）
    let _ = try_build("Q_jdouble_inline2x", &format!("{}\n__kernel void k(__global uint* o){{uint k[8]={{0,0,0,0,0,0,0,2}};uint Qx[8],Qy[8];uint RX[8],RY[8],RZ[8];uint one[8]={{0,0,0,0,0,0,0,1}};for(int i=0;i<NL;i++){{RX[i]=one[i];RY[i]=one[i];}}RZ[7]=0;int Rinf=1;for(int i=0;i<NL;i++)for(int bit=31;bit>=0;bit--){{uint kb=(k[i]>>bit)&1u;if(!Rinf){{uint A0[8],A1[8],A2[8];jdouble(A0,A1,A2,RX,RY,RZ);uint B0[8],B1[8],B2[8];jdouble(B0,B1,B2,A0,A1,A2);for(int t=0;t<NL;t++){{RX[t]=B0[t];RY[t]=B1[t];RZ[t]=B2[t];}}}}if(kb){{if(Rinf){{for(int t=0;t<NL;t++){{RX[t]=GX[t];RY[t]=GY[t];}}for(int t=0;t<NL;t++)RZ[t]=one[t];Rinf=0;}}else{{uint TX[8],TY[8],TZ[8];jdouble(TX,TY,TZ,RX,RY,RZ);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}}}for(int i=0;i<NL;i++){{Qx[i]=RX[i];Qy[i]=RY[i];}}}}\n", common));

    // R: jdouble 加 __attribute__((noinline))，scalar_mul 编译期循环调它（测 noinline 是否让 Radeon 接受）
    let common_r = common.replace(
        "static void jdouble(",
        "static void __attribute__((noinline)) jdouble(",
    );
    let _ = try_build("R_jdouble_noinline", &format!("{}\n__kernel void k(__global uint* o){{uint k[8]={{0,0,0,0,0,0,0,2}};uint Qx[8],Qy[8];uint RX[8],RY[8],RZ[8];uint one[8]={{0,0,0,0,0,0,0,1}};for(int i=0;i<NL;i++){{RX[i]=one[i];RY[i]=one[i];}}RZ[7]=0;int Rinf=1;for(int i=0;i<NL;i++)for(int bit=31;bit>=0;bit--){{uint kb=(k[i]>>bit)&1u;if(!Rinf){{uint TX[8],TY[8],TZ[8];jdouble(TX,TY,TZ,RX,RY,RZ);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}if(kb){{if(Rinf){{for(int t=0;t<NL;t++){{RX[t]=GX[t];RY[t]=GY[t];}}for(int t=0;t<NL;t++)RZ[t]=one[t];Rinf=0;}}else{{uint TX[8],TY[8],TZ[8];jadd_mixed(TX,TY,TZ,RX,RY,RZ,GX,GY);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}}}for(int i=0;i<NL;i++){{Qx[i]=RX[i];Qy[i]=RY[i];}}}}\n", common_r));

    // S: 同样的 noinline，但 jadd_mixed 也加 noinline
    let common_s = common_r.replace(
        "static void jadd_mixed(",
        "static void __attribute__((noinline)) jadd_mixed(",
    );
    let _ = try_build("S_both_noinline", &format!("{}\n__kernel void k(__global uint* o){{uint k[8]={{0,0,0,0,0,0,0,2}};uint Qx[8],Qy[8];uint RX[8],RY[8],RZ[8];uint one[8]={{0,0,0,0,0,0,0,1}};for(int i=0;i<NL;i++){{RX[i]=one[i];RY[i]=one[i];}}RZ[7]=0;int Rinf=1;for(int i=0;i<NL;i++)for(int bit=31;bit>=0;bit--){{uint kb=(k[i]>>bit)&1u;if(!Rinf){{uint TX[8],TY[8],TZ[8];jdouble(TX,TY,TZ,RX,RY,RZ);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}if(kb){{if(Rinf){{for(int t=0;t<NL;t++){{RX[t]=GX[t];RY[t]=GY[t];}}for(int t=0;t<NL;t++)RZ[t]=one[t];Rinf=0;}}else{{uint TX[8],TY[8],TZ[8];jadd_mixed(TX,TY,TZ,RX,RY,RZ,GX,GY);for(int t=0;t<NL;t++){{RX[t]=TX[t];RY[t]=TY[t];RZ[t]=TZ[t];}}}}}}for(int i=0;i<NL;i++){{Qx[i]=RX[i];Qy[i]=RY[i];}}}}\n", common_s));

    println!("=== diag done ===");
}
