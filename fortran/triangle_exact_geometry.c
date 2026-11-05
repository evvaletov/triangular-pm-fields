/* Exact predicates for the single active TAN_CACHE_V1 geometry.
 *
 * Process-global, like tan_mod's cache: callers must serialize loads, clears,
 * and queries. Not a separate per-thread cache and not thread safe. Cache
 * file plane indices are ONE based; all predicate API indices are ZERO based.
 * No query performs file I/O. No floating tolerance decides classification.
 *
 * tan_exact_load_c(path): 0 success, 1 I/O, 2 syntax/geometry, 3 size/allocation.
 * Every failed load clears the previous cache. Empty caches are valid.
 * Decimal tokens accept optional sign, decimal point, and e/E/d/D exponent.
 * NaN/Inf, hexadecimal, and rational slash forms are rejected. To bound input
 * resource use, decimal powers beyond +/-100000 are rejected (status 3),
 * never rounded. This exceeds all binary64 dyadic coordinate requirements.
 *
 * tan_exact_plane_c(i,p,raw,capacity): exact height sign -1/0/1, error 2.
 * raw is a NUL-terminated 55-significant-digit decimal approximation of the
 * UNNORMALIZED exact rational n.(p-A), suitable for real128 input. The sign
 * is decided before decimal conversion; exact zero is emitted as "0".
 * tan_exact_edge_c(i,p): 1 exact inclusive segment, 0 otherwise, -1 invalid.
 * tan_exact_support_c(i,p): 0 zero weighted winding, 1 nonzero, 2 surviving
 * boundary segment, -1 invalid/off-plane. Prepared edges must already have
 * coincident/overlapping boundary cancellation applied (cache contract).
 *
 * Build: gcc -std=c11 -O3 -fPIC -shared triangle_exact_geometry.c -lgmp.
 */
#define _POSIX_C_SOURCE 200809L
#include <gmp.h>
#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

typedef struct { mpq_t a[3], n[3]; } exact_plane;
typedef struct { int plane; mpq_t a[3], b[3], weight; } exact_edge;
static exact_plane *planes;
static exact_edge *edges;
static int nplanes, nedges, loaded;

void tan_exact_clear_c(void)
{
    for (int i=0; i<nplanes; ++i)
        for (int k=0; k<3; ++k) {
            mpq_clear(planes[i].a[k]); mpq_clear(planes[i].n[k]);
        }
    for (int i=0; i<nedges; ++i) {
        for (int k=0; k<3; ++k) {
            mpq_clear(edges[i].a[k]); mpq_clear(edges[i].b[k]);
        }
        mpq_clear(edges[i].weight);
    }
    free(planes); free(edges);
    planes=NULL; edges=NULL; nplanes=nedges=loaded=0;
}

/* Parse decimal text directly into an exact rational, never through atof. */
static int decimal(mpq_t q, const char *s)
{
    size_t len=strlen(s), digits=0, fractional=0;
    char *whole=malloc(len+2);
    if (!whole) return 3;
    const char *p=s;
    int negative=0, dot=0, status=2;
    if (*p=='+' || *p=='-') { negative=(*p=='-'); ++p; }
    for (; *p; ++p) {
        if (*p>='0' && *p<='9') {
            whole[digits++]=*p;
            if (dot) ++fractional;
        } else if (*p=='.' && !dot) dot=1;
        else break;
    }
    if (!digits) goto done;
    long exponent=0;
    if (*p=='e' || *p=='E' || *p=='d' || *p=='D') {
        char *end;
        ++p;
        if (!*p) goto done;
        errno=0;
        exponent=strtol(p,&end,10);
        if (errno==ERANGE) { status=3; goto done; }
        if (end==p || *end) goto done;
        p=end;
    }
    if (*p) goto done;
    if (fractional>100000 || exponent>100000 || exponent< -100000) {
        status=3; goto done;
    }
    exponent-=(long)fractional;
    if (exponent>100000 || exponent< -100000) { status=3; goto done; }
    whole[digits]='\0';
    if (mpz_set_str(mpq_numref(q),whole,10)) goto done;
    mpz_set_ui(mpq_denref(q),1);
    if (negative) mpz_neg(mpq_numref(q),mpq_numref(q));
    if (mpq_sgn(q) && exponent) {
        mpz_t power;
        mpz_init(power);
        mpz_ui_pow_ui(power,10,(unsigned long)(exponent<0 ? -exponent : exponent));
        if (exponent<0) mpz_set(mpq_denref(q),power);
        else mpz_mul(mpq_numref(q),mpq_numref(q),power);
        mpz_clear(power);
    }
    mpq_canonicalize(q);
    status=0;
done:
    free(whole);
    return status;
}

static int tokens(char *line, char **out, int capacity)
{
    int n=0;
    char *save=NULL;
    for (char *p=strtok_r(line," \t\r\n",&save); p;
         p=strtok_r(NULL," \t\r\n",&save)) {
        if (n==capacity) return -1;
        out[n++]=p;
    }
    return n;
}

static int natural(const char *s, int *value)
{
    char *end;
    if (!*s || *s=='-') return 2;
    errno=0;
    unsigned long v=strtoul(s,&end,10);
    if (errno || end==s || *end || v>INT_MAX) return 2;
    *value=(int)v;
    return 0;
}

static int read_line(FILE *file, char **line, size_t *capacity)
{
    ssize_t length=getline(line,capacity,file);
    if (length<0) return 0;
    /* Embedded NUL must not hide an invalid suffix from token parsing. */
    return memchr(*line,'\0',(size_t)length) ? -1 : 1;
}

static void height(mpq_t out, const exact_plane *plane, mpq_t p[3])
{
    mpq_t t;
    mpq_init(t);
    mpq_set_ui(out,0,1);
    for (int k=0; k<3; ++k) {
        mpq_sub(t,p[k],plane->a[k]);
        mpq_mul(t,t,plane->n[k]);
        mpq_add(out,out,t);
    }
    mpq_clear(t);
}

int tan_exact_load_c(const char *path)
{
    tan_exact_clear_c();
    if (!path || !*path) return 1;
    FILE *file=fopen(path,"r");
    if (!file) return 1;
    char *line=NULL, *words[8];
    size_t capacity=0;
    int status=2, np=0, ne=0;
    if (read_line(file,&line,&capacity)!=1 || tokens(line,words,8)!=1 ||
        strcmp(words[0],"TAN_CACHE_V1")) goto done;
    if (read_line(file,&line,&capacity)!=1 || tokens(line,words,8)!=2 ||
        natural(words[0],&np) || natural(words[1],&ne)) goto done;
    /* Reject impossible counts before allocating, using only the minimum
     * serialized record lengths; this imposes no geometry-size cutoff. */
    struct stat info;
    off_t offset=ftello(file);
    if (offset>=0 && !fstat(fileno(file),&info) && S_ISREG(info.st_mode)) {
        uint64_t minimum=(uint64_t)np*11+(uint64_t)ne*15;
        if (info.st_size<offset || (uint64_t)(info.st_size-offset)<minimum)
            goto done;
    }
    if ((size_t)np>SIZE_MAX/sizeof(*planes) ||
        (size_t)ne>SIZE_MAX/sizeof(*edges)) { status=3; goto done; }
    planes=calloc((size_t)np,sizeof(*planes));
    edges=calloc((size_t)ne,sizeof(*edges));
    if ((np && !planes) || (ne && !edges)) { status=3; goto done; }
    for (int i=0; i<np; ++i) {
        for (int k=0; k<3; ++k) {
            mpq_init(planes[i].a[k]); mpq_init(planes[i].n[k]);
        }
        ++nplanes;
    }
    for (int i=0; i<ne; ++i) {
        for (int k=0; k<3; ++k) {
            mpq_init(edges[i].a[k]); mpq_init(edges[i].b[k]);
        }
        mpq_init(edges[i].weight);
        ++nedges;
    }
    for (int i=0; i<np; ++i) {
        status=2;
        if (read_line(file,&line,&capacity)!=1 || tokens(line,words,8)!=6) goto done;
        for (int k=0; k<3; ++k) {
            if ((status=decimal(planes[i].a[k],words[k]))) goto done;
            if ((status=decimal(planes[i].n[k],words[k+3]))) goto done;
        }
        if (!mpq_sgn(planes[i].n[0]) && !mpq_sgn(planes[i].n[1]) &&
            !mpq_sgn(planes[i].n[2])) { status=2; goto done; }
    }
    for (int i=0; i<ne; ++i) {
        status=2;
        int index;
        if (read_line(file,&line,&capacity)!=1 || tokens(line,words,8)!=8 ||
            natural(words[0],&index) || index<1 || index>np) goto done;
        edges[i].plane=index-1;
        for (int k=0; k<3; ++k) {
            if ((status=decimal(edges[i].a[k],words[k+1]))) goto done;
            if ((status=decimal(edges[i].b[k],words[k+4]))) goto done;
        }
        if ((status=decimal(edges[i].weight,words[7]))) goto done;
        int different=0;
        for (int k=0; k<3; ++k)
            different |= mpq_cmp(edges[i].a[k],edges[i].b[k])!=0;
        mpq_t h;
        mpq_init(h);
        height(h,&planes[index-1],edges[i].a);
        int onplane=!mpq_sgn(h);
        height(h,&planes[index-1],edges[i].b);
        onplane=onplane && !mpq_sgn(h);
        mpq_clear(h);
        if (!different || !onplane) { status=2; goto done; }
    }
    status=2;
    int record;
    while ((record=read_line(file,&line,&capacity))==1)
        if (tokens(line,words,8)!=0) goto done;
    if (record<0) goto done;
    if (ferror(file)) { status=1; goto done; }
    status=0;
done:
    free(line);
    if (ferror(file)) status=1;
    if (fclose(file) && !status) status=1;
    if (status) tan_exact_clear_c();
    else loaded=1;
    return status;
}

static int point_init(mpq_t exact[3], const double p[3])
{
    if (!p || !isfinite(p[0]) || !isfinite(p[1]) || !isfinite(p[2])) return 0;
    for (int k=0; k<3; ++k) { mpq_init(exact[k]); mpq_set_d(exact[k],p[k]); }
    return 1;
}

static void point_clear(mpq_t p[3])
{
    for (int k=0; k<3; ++k) mpq_clear(p[k]);
}

int tan_exact_plane_c(int plane0, const double p[3], char *raw_height, int capacity)
{
    if (raw_height && capacity>0) raw_height[0]='\0';
    if (!loaded || plane0<0 || plane0>=nplanes || !raw_height || capacity<2) return 2;
    mpq_t q[3], h;
    if (!point_init(q,p)) return 2;
    mpq_init(h);
    height(h,&planes[plane0],q);
    int sign=mpq_sgn(h), count;
    if (!sign) count=snprintf(raw_height,(size_t)capacity,"0");
    else {
        mpf_t f;
        mpf_init2(f,256);
        mpf_set_q(f,h);
        mp_exp_t exponent;
        char *mantissa=mpf_get_str(NULL,&exponent,10,55,f);
        const char *digits=mantissa+(mantissa[0]=='-');
        count=snprintf(raw_height,(size_t)capacity,"%s%c.%se%+" PRIdMAX,
                       sign<0 ? "-" : "",digits[0],digits+1,(intmax_t)exponent-1);
        void (*gmp_free)(void *,size_t);
        mp_get_memory_functions(NULL,NULL,&gmp_free);
        gmp_free(mantissa,strlen(mantissa)+1);
        mpf_clear(f);
    }
    mpq_clear(h);
    point_clear(q);
    if (count<0 || count>=capacity) { raw_height[0]='\0'; return 2; }
    return sign;
}

static int on_segment(const exact_edge *edge, mpq_t p[3])
{
    mpq_t d[3], r[3], x, y;
    for (int k=0; k<3; ++k) {
        mpq_init(d[k]); mpq_init(r[k]);
        mpq_sub(d[k],edge->b[k],edge->a[k]);
        mpq_sub(r[k],p[k],edge->a[k]);
    }
    mpq_init(x); mpq_init(y);
    int result=1;
    for (int k=0; k<3; ++k) {
        int j=(k+1)%3;
        mpq_mul(x,d[k],r[j]); mpq_mul(y,d[j],r[k]);
        if (mpq_cmp(x,y)) result=0;
        int ca=mpq_cmp(p[k],edge->a[k]), cb=mpq_cmp(p[k],edge->b[k]);
        if ((ca<0 && cb<0) || (ca>0 && cb>0)) result=0;
    }
    point_clear(d); point_clear(r); mpq_clear(x); mpq_clear(y);
    return result;
}

int tan_exact_edge_c(int edge0, const double p[3])
{
    if (!loaded || edge0<0 || edge0>=nedges) return -1;
    mpq_t q[3];
    if (!point_init(q,p)) return -1;
    int result=on_segment(&edges[edge0],q);
    point_clear(q);
    return result;
}

int tan_exact_support_c(int plane0, const double p[3])
{
    if (!loaded || plane0<0 || plane0>=nplanes) return -1;
    mpq_t q[3], h, winding, dx, dy, rx, ry, cross, temp;
    if (!point_init(q,p)) return -1;
    mpq_inits(h,winding,dx,dy,rx,ry,cross,temp,NULL);
    height(h,&planes[plane0],q);
    int result=-1;
    if (mpq_sgn(h)) goto done;
    int drop=0;
    while (!mpq_sgn(planes[plane0].n[drop])) ++drop;
    int x=(drop+1)%3, y=(drop+2)%3;
    for (int i=0; i<nedges; ++i) {
        exact_edge *edge=&edges[i];
        if (edge->plane!=plane0 || !mpq_sgn(edge->weight)) continue;
        if (on_segment(edge,q)) { result=2; goto done; }
        mpq_sub(dx,edge->b[x],edge->a[x]);
        mpq_sub(dy,edge->b[y],edge->a[y]);
        mpq_sub(rx,q[x],edge->a[x]);
        mpq_sub(ry,q[y],edge->a[y]);
        mpq_mul(cross,dx,ry); mpq_mul(temp,dy,rx); mpq_sub(cross,cross,temp);
        int a=mpq_cmp(edge->a[y],q[y]), b=mpq_cmp(edge->b[y],q[y]);
        if (a<=0 && b>0 && mpq_sgn(cross)>0)
            mpq_add(winding,winding,edge->weight);
        else if (b<=0 && a>0 && mpq_sgn(cross)<0)
            mpq_sub(winding,winding,edge->weight);
    }
    result=mpq_sgn(winding)!=0;
done:
    mpq_clears(h,winding,dx,dy,rx,ry,cross,temp,NULL);
    point_clear(q);
    return result;
}
