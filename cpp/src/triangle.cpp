#include "triangle_pm/triangle.hpp"

#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/cpp_int.hpp>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <mutex>

namespace triangle_pm {
namespace {

using Real = boost::multiprecision::cpp_dec_float_100;
using Rational = boost::multiprecision::cpp_rational;
using Integer = boost::multiprecision::cpp_int;

constexpr int kNpair = 924;

int nmono_of[kMaxOrder + 1];
int expo_x[kMaxMono];
int expo_y[kMaxMono];
int expo_z[kMaxMono];
int expo_deg[kMaxMono];
int idx_of[kMaxOrder + 1][kMaxOrder + 1][kMaxOrder + 1];
int pair_i[kNpair];
int pair_j[kNpair];
int pair_k[kNpair];
int npair_of[kMaxOrder + 1];
std::once_flag tables_once;

void init_tables() {
  std::call_once(tables_once, [] {
  int k = 0;
  for (int q = 0; q <= kMaxOrder; ++q) {
    for (int a = 0; a <= q; ++a) {
      for (int b = 0; b <= q - a; ++b) {
        const int z = q - a - b;
        expo_x[k] = a;
        expo_y[k] = b;
        expo_z[k] = z;
        expo_deg[k] = q;
        idx_of[a][b][z] = k;
        ++k;
      }
    }
    nmono_of[q] = k;
  }
  int n = 0;
  for (int dsum = 0; dsum <= kMaxOrder; ++dsum) {
    for (int i = 0; i < kMaxMono; ++i) {
      if (expo_deg[i] > dsum) continue;
      for (int j = 0; j < kMaxMono; ++j) {
        if (expo_deg[j] != dsum - expo_deg[i]) continue;
        pair_i[n] = i;
        pair_j[n] = j;
        pair_k[n] = idx_of[expo_x[i] + expo_x[j]][expo_y[i] + expo_y[j]]
                          [expo_z[i] + expo_z[j]];
        ++n;
      }
    }
    npair_of[dsum] = n;
  }
  });
}

bool finite_d(double x) { return std::isfinite(x); }

bool parse_finite_or_any(const std::string& token, double& value) {
  if (token.empty()) return false;
  char* end = nullptr;
  errno = 0;
  value = std::strtod(token.c_str(), &end);
  return end != token.c_str() && *end == '\0';
}

Rational rational_from_binary64(double x, bool& ok) {
  ok = false;
  if (!finite_d(x)) return Rational(0);
  std::uint64_t bits = 0;
  std::memcpy(&bits, &x, sizeof(bits));
  const int sign = static_cast<int>((bits >> 63) & 1u);
  const int exp = static_cast<int>((bits >> 52) & 0x7ffu);
  const std::uint64_t frac = bits & 0xfffffffffffffULL;
  Rational r;
  if (exp == 0) {
    if (frac == 0) {
      ok = true;
      return Rational(0);
    }
    r = Rational(Integer(frac), Integer(1) << 1074);
  } else if (exp == 0x7ff) {
    return Rational(0);
  } else {
    const Integer mant = (Integer(1) << 52) + Integer(frac);
    const int power = exp - 1023 - 52;
    if (power >= 0) {
      r = Rational(mant << power);
    } else {
      r = Rational(mant, Integer(1) << (-power));
    }
  }
  if (sign) r = -r;
  const double back = r.convert_to<double>();
  ok = (back == x);
  return r;
}

struct Rat3 {
  Rational x, y, z;
};

Rat3 rat3_from_binary64(const std::array<double, 3>& v, bool& ok) {
  bool ox = false, oy = false, oz = false;
  Rat3 r{rational_from_binary64(v[0], ox), rational_from_binary64(v[1], oy),
         rational_from_binary64(v[2], oz)};
  ok = ox && oy && oz;
  return r;
}

Rat3 sub(const Rat3& a, const Rat3& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

Rat3 cross(const Rat3& a, const Rat3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

Rational dot(const Rat3& a, const Rat3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

bool is_zero(const Rat3& a) { return a.x == 0 && a.y == 0 && a.z == 0; }

bool equal(const Rat3& a, const Rat3& b) {
  return a.x == b.x && a.y == b.y && a.z == b.z;
}

Rational coord(const Rat3& a, int k) { return k == 0 ? a.x : (k == 1 ? a.y : a.z); }

bool on_segment(const Rat3& a, const Rat3& b, const Rat3& p) {
  if (!is_zero(cross(sub(b, a), sub(p, a)))) return false;
  for (int k = 0; k < 3; ++k) {
    const Rational pk = coord(p, k);
    const Rational ak = coord(a, k);
    const Rational bk = coord(b, k);
    if ((pk < ak && pk < bk) || (pk > ak && pk > bk)) return false;
  }
  return true;
}

enum class Place { Regular, Vertex, Edge, Interior, ExteriorPlane };

Place classify(const Rat3& a, const Rat3& b, const Rat3& c, const Rat3& p,
               const Rat3& nraw) {
  if (equal(p, a) || equal(p, b) || equal(p, c)) return Place::Vertex;
  if (on_segment(a, b, p) || on_segment(b, c, p) || on_segment(c, a, p)) {
    return Place::Edge;
  }
  if (dot(sub(p, a), nraw) != 0) return Place::Regular;
  const Rational alpha = dot(cross(sub(c, b), sub(p, b)), nraw);
  const Rational beta = dot(cross(sub(a, c), sub(p, c)), nraw);
  const Rational gamma = dot(cross(sub(b, a), sub(p, a)), nraw);
  if (alpha > 0 && beta > 0 && gamma > 0) return Place::Interior;
  return Place::ExteriorPlane;
}

struct Jet {
  int q = 0;
  Real c[kMaxMono]{};
};

Jet jet_zero(int q) {
  Jet j;
  j.q = q;
  return j;
}

Jet jet_const(const Real& v, int q) {
  Jet j = jet_zero(q);
  j.c[0] = v;
  return j;
}

Jet jet_coord(const Real& center, int axis, const Real& radius, int q) {
  Jet j = jet_const(center, q);
  if (q <= 0) return j;
  int ix = 0, iy = 0, iz = 0;
  if (axis == 0) ix = 1;
  if (axis == 1) iy = 1;
  if (axis == 2) iz = 1;
  j.c[idx_of[ix][iy][iz]] = radius;
  return j;
}

Jet jet_add(const Jet& a, const Jet& b) {
  Jet c = jet_zero(a.q);
  const int n = nmono_of[a.q];
  for (int i = 0; i < n; ++i) c.c[i] = a.c[i] + b.c[i];
  return c;
}

Jet jet_sub(const Jet& a, const Jet& b) {
  Jet c = jet_zero(a.q);
  const int n = nmono_of[a.q];
  for (int i = 0; i < n; ++i) c.c[i] = a.c[i] - b.c[i];
  return c;
}

Jet jet_neg(const Jet& a) {
  Jet c = jet_zero(a.q);
  const int n = nmono_of[a.q];
  for (int i = 0; i < n; ++i) c.c[i] = -a.c[i];
  return c;
}

Jet jet_scale(const Jet& a, const Real& s) {
  Jet c = jet_zero(a.q);
  const int n = nmono_of[a.q];
  for (int i = 0; i < n; ++i) c.c[i] = s * a.c[i];
  return c;
}

Jet jet_mul(const Jet& a, const Jet& b) {
  Jet c = jet_zero(a.q);
  const int np = npair_of[a.q];
  for (int p = 0; p < np; ++p) {
    const Real ai = a.c[pair_i[p]];
    const Real bj = b.c[pair_j[p]];
    if (ai == 0 || bj == 0) continue;
    c.c[pair_k[p]] += ai * bj;
  }
  return c;
}

bool jet_finite(const Jet& a) {
  const int n = nmono_of[a.q];
  for (int i = 0; i < n; ++i) {
    if (!boost::multiprecision::isfinite(a.c[i])) return false;
  }
  return true;
}

Jet jet_pow(const Jet& a, const Real& exponent, int& status) {
  const Real a0 = a.c[0];
  if (a0 == 0) {
    status = kInvalidGeometry;
    return jet_zero(a.q);
  }
  if (exponent == 0) return jet_const(Real(1), a.q);
  if (exponent == 1) return a;
  Jet u = jet_scale(jet_sub(a, jet_const(a0, a.q)), Real(1) / a0);
  Jet total = jet_const(Real(1), a.q);
  Jet term = jet_const(Real(1), a.q);
  Real coeff = 1;
  for (int k = 1; k <= a.q; ++k) {
    term = jet_mul(term, u);
    coeff *= (exponent - Real(k) + 1) / Real(k);
    total = jet_add(total, jet_scale(term, coeff));
  }
  return jet_scale(total, boost::multiprecision::pow(a0, exponent));
}

Jet jet_inv(const Jet& a, int& status) { return jet_pow(a, Real(-1), status); }

Jet jet_sqrt(const Jet& a, int& status) {
  if (a.c[0] < 0) {
    status = kInvalidGeometry;
    return jet_zero(a.q);
  }
  return jet_pow(a, Real(1) / 2, status);
}

Jet jet_log(const Jet& a, int& status) {
  const Real a0 = a.c[0];
  if (a0 <= 0) {
    status = kInvalidGeometry;
    return jet_zero(a.q);
  }
  Jet u = jet_scale(jet_sub(a, jet_const(a0, a.q)), Real(1) / a0);
  Jet total = jet_const(boost::multiprecision::log(a0), a.q);
  Jet term = jet_const(Real(1), a.q);
  Real s = 1;
  for (int k = 1; k <= a.q; ++k) {
    term = jet_mul(term, u);
    s = -s;
    total = jet_add(total, jet_scale(term, -s / Real(k)));
  }
  return total;
}

// atan2(y,x) = atan2(y0,x0) + atan(t), t=(x0 y - y0 x)/(x0 x + y0 y), t0=0.
Jet jet_atan2(const Jet& y, const Jet& x, int& status) {
  const Real x0 = x.c[0];
  const Real y0 = y.c[0];
  const Real r2 = x0 * x0 + y0 * y0;
  if (r2 == 0) {
    status = kInvalidGeometry;
    return jet_zero(x.q);
  }
  Jet num = jet_sub(jet_scale(y, x0), jet_scale(x, y0));
  Jet den = jet_add(jet_scale(x, x0), jet_scale(y, y0));
  if (den.c[0] == 0) {
    status = kInvalidGeometry;
    return jet_zero(x.q);
  }
  int st = kOk;
  Jet t = jet_mul(num, jet_inv(den, st));
  if (st != kOk) {
    status = st;
    return jet_zero(x.q);
  }
  t.c[0] = 0;
  Jet odd = t;
  Jet atan_t = jet_zero(x.q);
  Real sign = 1;
  for (int n = 0; 2 * n + 1 <= x.q; ++n) {
    atan_t = jet_add(atan_t, jet_scale(odd, sign / Real(2 * n + 1)));
    sign = -sign;
    odd = jet_mul(jet_mul(odd, t), t);
  }
  Real omega0 = boost::multiprecision::atan2(y0, x0);
  if (y0 == 0) omega0 = 0;
  return jet_add(jet_const(omega0, x.q), atan_t);
}

struct Jet3 {
  Jet x, y, z;
};

Jet3 jet3_const(const Real& ax, const Real& ay, const Real& az, int q) {
  return {jet_const(ax, q), jet_const(ay, q), jet_const(az, q)};
}

Jet3 jet3_point(const Real& px, const Real& py, const Real& pz, const Real& radius,
                int q) {
  return {jet_coord(px, 0, radius, q), jet_coord(py, 1, radius, q),
          jet_coord(pz, 2, radius, q)};
}

Jet3 jet3_sub(const Jet3& a, const Jet3& b) {
  return {jet_sub(a.x, b.x), jet_sub(a.y, b.y), jet_sub(a.z, b.z)};
}

Jet3 jet3_scale(const Jet3& a, const Real& s) {
  return {jet_scale(a.x, s), jet_scale(a.y, s), jet_scale(a.z, s)};
}

Jet jet3_dot(const Jet3& a, const Jet3& b) {
  return jet_add(jet_add(jet_mul(a.x, b.x), jet_mul(a.y, b.y)), jet_mul(a.z, b.z));
}

Jet3 jet3_cross(const Jet3& a, const Jet3& b) {
  return {jet_sub(jet_mul(a.y, b.z), jet_mul(a.z, b.y)),
          jet_sub(jet_mul(a.z, b.x), jet_mul(a.x, b.z)),
          jet_sub(jet_mul(a.x, b.y), jet_mul(a.y, b.x))};
}

Jet jet3_norm2(const Jet3& a) { return jet3_dot(a, a); }

Jet jet3_norm(const Jet3& a, int& status) { return jet_sqrt(jet3_norm2(a), status); }

const Real kPi = boost::multiprecision::atan(Real(1)) * 4;
const Real kFourPi = kPi * 4;

Result fail(int status) {
  Result r;
  r.status = status;
  return r;
}

Result success_zero(int order) {
  Result r;
  r.status = kOk;
  r.exponents = monomials(order);
  const int n = nmono(order);
  r.coefficients[0].assign(n, 0.0);
  r.coefficients[1].assign(n, 0.0);
  r.coefficients[2].assign(n, 0.0);
  return r;
}

bool to_double(const Real& x, double& out) {
  if (!boost::multiprecision::isfinite(x)) return false;
  out = x.convert_to<double>();
  return finite_d(out);
}

int evaluate_field(int order, const Real& radius, const Real& sigma,
                   const Real va[3], const Real vb[3], const Real vc[3],
                   const Real p0[3], bool on_plane, Jet& bx, Jet& by, Jet& bz) {
  const int q = order;
  Real e1[3] = {vb[0] - va[0], vb[1] - va[1], vb[2] - va[2]};
  Real e2[3] = {vc[0] - va[0], vc[1] - va[1], vc[2] - va[2]};
  Real nraw[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                  e1[0] * e2[1] - e1[1] * e2[0]};
  const Real n2 =
      nraw[0] * nraw[0] + nraw[1] * nraw[1] + nraw[2] * nraw[2];
  if (n2 == 0) return kInvalidGeometry;
  const Real nmag = boost::multiprecision::sqrt(n2);
  Real nh[3] = {nraw[0] / nmag, nraw[1] / nmag, nraw[2] / nmag};

  Jet3 p = jet3_point(p0[0], p0[1], p0[2], radius, q);
  const Real verts[3][3] = {{va[0], va[1], va[2]},
                            {vb[0], vb[1], vb[2]},
                            {vc[0], vc[1], vc[2]}};
  Jet3 rv[3];
  Jet rho[3];
  int st = kOk;
  for (int i = 0; i < 3; ++i) {
    rv[i] = jet3_sub(jet3_const(verts[i][0], verts[i][1], verts[i][2], q), p);
    rho[i] = jet3_norm(rv[i], st);
    if (st != kOk) return kPhysicalEdgeVertex;
    if (rho[i].c[0] == 0) return kPhysicalEdgeVertex;
  }

  Jet3 nrawj = jet3_const(nraw[0], nraw[1], nraw[2], q);
  Jet num = jet3_dot(rv[0], nrawj);
  if (on_plane) num.c[0] = 0;

  Jet den = jet_mul(jet_mul(rho[0], rho[1]), rho[2]);
  den = jet_add(den, jet_mul(jet3_dot(rv[0], rv[1]), rho[2]));
  den = jet_add(den, jet_mul(jet3_dot(rv[0], rv[2]), rho[1]));
  den = jet_add(den, jet_mul(jet3_dot(rv[1], rv[2]), rho[0]));

  Jet omega;
  if (on_plane && q == 0) {
    omega = jet_zero(q);
  } else {
    if (num.c[0] == 0 && den.c[0] == 0) return kInvalidGeometry;
    omega = jet_atan2(num, den, st);
    if (st != kOk) return st;
    omega = jet_scale(omega, Real(2));
    if (on_plane) omega.c[0] = 0;
  }

  const Real fac = sigma / kFourPi;
  Jet3 nhj = jet3_const(nh[0], nh[1], nh[2], q);
  bx = jet_scale(jet_mul(omega, nhj.x), -fac);
  by = jet_scale(jet_mul(omega, nhj.y), -fac);
  bz = jet_scale(jet_mul(omega, nhj.z), -fac);

  for (int i = 0; i < 3; ++i) {
    const int j = (i + 1) % 3;
    Real chord[3] = {verts[j][0] - verts[i][0], verts[j][1] - verts[i][1],
                     verts[j][2] - verts[i][2]};
    const Real length = boost::multiprecision::sqrt(
        chord[0] * chord[0] + chord[1] * chord[1] + chord[2] * chord[2]);
    if (length == 0) return kInvalidGeometry;
    Real t[3] = {chord[0] / length, chord[1] / length, chord[2] / length};
    Jet3 tj = jet3_const(t[0], t[1], t[2], q);
    Jet3 chordj = jet3_const(chord[0], chord[1], chord[2], q);
    Jet u = jet_neg(jet3_dot(rv[i], tj));
    Jet w = jet3_dot(rv[j], tj);
    const Real u0 = u.c[0];
    const Real w0 = w.c[0];
    Jet3 offset = (boost::multiprecision::abs(u0) <= boost::multiprecision::abs(w0))
                      ? jet3_scale(rv[i], Real(-1))
                      : jet3_scale(rv[j], Real(-1));
    Jet3 cr = jet3_cross(offset, chordj);
    Jet rho2 = jet_scale(jet3_norm2(cr), Real(1) / (length * length));
    const int su = (u0 < 0) ? -1 : 1;
    const int sw = (w0 < 0) ? -1 : 1;
    Jet tmp1 = jet_add(rho[i], jet_scale(u, Real(su)));
    Jet tmp2 = jet_add(rho[j], jet_scale(w, Real(sw)));
    if (tmp1.c[0] == 0 || tmp2.c[0] == 0) return kPhysicalEdgeVertex;
    tmp1 = jet_inv(tmp1, st);
    if (st != kOk) return st;
    tmp2 = jet_inv(tmp2, st);
    if (st != kOk) return st;
    Jet gap = jet_add(jet_scale(u, Real(su - 1)), jet_scale(w, Real(sw - 1)));
    gap = jet_add(gap, jet_mul(rho2, jet_add(tmp1, tmp2)));
    if (gap.c[0] <= 0) return kInvalidGeometry;
    Jet invgap = jet_inv(gap, st);
    if (st != kOk) return st;
    Jet L = jet_log(jet_add(jet_const(Real(1), q), jet_scale(invgap, Real(2) * length)),
                    st);
    if (st != kOk) return st;
    Real nxt[3] = {nh[1] * t[2] - nh[2] * t[1], nh[2] * t[0] - nh[0] * t[2],
                   nh[0] * t[1] - nh[1] * t[0]};
    bx = jet_add(bx, jet_scale(jet_mul(L, jet_const(nxt[0], q)), -fac));
    by = jet_add(by, jet_scale(jet_mul(L, jet_const(nxt[1], q)), -fac));
    bz = jet_add(bz, jet_scale(jet_mul(L, jet_const(nxt[2], q)), -fac));
  }

  if (!jet_finite(bx) || !jet_finite(by) || !jet_finite(bz)) return kNonfiniteResult;
  return kOk;
}

}  // namespace

std::vector<std::array<int, 3>> monomials(int order) {
  init_tables();
  std::vector<std::array<int, 3>> out;
  if (order < 0 || order > kMaxOrder) return out;
  const int n = nmono_of[order];
  out.resize(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) out[static_cast<std::size_t>(i)] = {expo_x[i], expo_y[i], expo_z[i]};
  return out;
}

bool parse_binary64(const std::string& token, double& value) {
  return parse_finite_or_any(token, value);
}

Result evaluate(int order, double radius, double sigma,
                const std::array<double, 3>& a, const std::array<double, 3>& b,
                const std::array<double, 3>& c, const std::array<double, 3>& p) {
  init_tables();
  if (order < 0 || order > kMaxOrder) return fail(kInvalidOrder);
  if (!finite_d(p[0]) || !finite_d(p[1]) || !finite_d(p[2])) {
    return fail(kNonfiniteObserver);
  }
  if (!finite_d(radius) || radius <= 0.0) return fail(kBadRadius);
  if (!finite_d(sigma) || !finite_d(a[0]) || !finite_d(a[1]) || !finite_d(a[2]) ||
      !finite_d(b[0]) || !finite_d(b[1]) || !finite_d(b[2]) || !finite_d(c[0]) ||
      !finite_d(c[1]) || !finite_d(c[2])) {
    return fail(kInvalidGeometry);
  }

  bool ok_a = false, ok_b = false, ok_c = false, ok_p = false, ok_s = false;
  const Rat3 ra = rat3_from_binary64(a, ok_a);
  const Rat3 rb = rat3_from_binary64(b, ok_b);
  const Rat3 rc = rat3_from_binary64(c, ok_c);
  const Rat3 rp = rat3_from_binary64(p, ok_p);
  const Rational rsigma = rational_from_binary64(sigma, ok_s);
  if (!ok_a || !ok_b || !ok_c || !ok_p || !ok_s) return fail(kInvalidGeometry);

  const Rat3 nraw = cross(sub(rb, ra), sub(rc, ra));
  if (is_zero(nraw)) return fail(kInvalidGeometry);

  const Place place = classify(ra, rb, rc, rp, nraw);
  if (rsigma == 0) return success_zero(order);

  if (place == Place::Vertex || place == Place::Edge) return fail(kPhysicalEdgeVertex);
  if (place == Place::Interior && order > 0) return fail(kDaOnChargedSheet);

  const Real Rva[3] = {Real(a[0]), Real(a[1]), Real(a[2])};
  const Real Rvb[3] = {Real(b[0]), Real(b[1]), Real(b[2])};
  const Real Rvc[3] = {Real(c[0]), Real(c[1]), Real(c[2])};
  const Real Rp[3] = {Real(p[0]), Real(p[1]), Real(p[2])};
  Jet bx, by, bz;
  const int st =
      evaluate_field(order, Real(radius), Real(sigma), Rva, Rvb, Rvc, Rp,
                     place == Place::Interior || place == Place::ExteriorPlane, bx, by,
                     bz);
  if (st != kOk) return fail(st);

  Result r;
  r.status = kOk;
  r.exponents = monomials(order);
  const int n = nmono(order);
  r.coefficients[0].resize(static_cast<std::size_t>(n));
  r.coefficients[1].resize(static_cast<std::size_t>(n));
  r.coefficients[2].resize(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    if (!to_double(bx.c[i], r.coefficients[0][static_cast<std::size_t>(i)]) ||
        !to_double(by.c[i], r.coefficients[1][static_cast<std::size_t>(i)]) ||
        !to_double(bz.c[i], r.coefficients[2][static_cast<std::size_t>(i)])) {
      return fail(kNonfiniteResult);
    }
  }
  return r;
}

Result evaluate_tokens(const std::vector<std::string>& tokens) {
  if (tokens.size() != 15) return fail(kInvalidGeometry);
  double vals[15];
  for (int i = 0; i < 15; ++i) {
    if (!parse_binary64(tokens[static_cast<std::size_t>(i)], vals[i])) {
      return fail(kInvalidGeometry);
    }
  }
  const double order_d = vals[0];
  if (!finite_d(order_d) || order_d != std::floor(order_d) || order_d < 0.0 ||
      order_d > static_cast<double>(kMaxOrder)) {
    return fail(kInvalidOrder);
  }
  const int order = static_cast<int>(order_d);
  const double radius = vals[1];
  const double sigma = vals[2];
  const std::array<double, 3> a{vals[3], vals[4], vals[5]};
  const std::array<double, 3> b{vals[6], vals[7], vals[8]};
  const std::array<double, 3> c{vals[9], vals[10], vals[11]};
  const std::array<double, 3> p{vals[12], vals[13], vals[14]};
  return evaluate(order, radius, sigma, a, b, c, p);
}

std::string to_json(const Result& result) {
  std::ostringstream os;
  os.setf(std::ios::fmtflags(0), std::ios::floatfield);
  os << "{\"status\":" << result.status << ",\"exponents\":[";
  if (result.status == kOk) {
    for (std::size_t i = 0; i < result.exponents.size(); ++i) {
      if (i) os << ',';
      os << '[' << result.exponents[i][0] << ',' << result.exponents[i][1] << ','
         << result.exponents[i][2] << ']';
    }
  }
  os << "],\"coefficients\":";
  if (result.status != kOk) {
    os << "[]}";
    return os.str();
  }
  os << '[';
  for (int comp = 0; comp < 3; ++comp) {
    if (comp) os << ',';
    os << '[';
    for (std::size_t i = 0; i < result.coefficients[comp].size(); ++i) {
      if (i) os << ',';
      const double x = result.coefficients[comp][i];
      char buf[64];
      std::snprintf(buf, sizeof(buf), "%.16e", x);
      os << buf;
    }
    os << ']';
  }
  os << "]}";
  return os.str();
}

}  // namespace triangle_pm
