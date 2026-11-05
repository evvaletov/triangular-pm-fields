"""
TriangularPMFields — standalone charged-triangle μ0 H and a native Fortran client.

Standalone code evaluates the surface integral
μ0 H = σ/(4π) ∫_T (p-s)/|p-s|^3 dS
with exact binary64 rational membership and scoped high-precision Taylor
arithmetic. It does not call Python, Fortran, or another evaluator.

The native client (native_cli.jl / native_*) shares the Fortran tan_mod
physical kernel through the documented C ABI. That path is a separately
named interface.

Limits: orders 0..6, truncated local series, corpus-bounded accuracy.
Radius scales monomials; it is not a convergence certificate. Membership
uses exact binary64 rationals, not a floating tolerance. The native cache
is process-global and is not thread-safe. Scoped BigFloat precision also
requires serial standalone calls; use separate processes for concurrent work.
"""
module TriangularPMFields

import Libdl

export nmono, exponents, evaluate_map, triangle_map, FieldResult
export write_result_json, eval_standalone_tokens
export native_eval, native_load!, native_last_mode
export STATUS_OK, STATUS_ORDER, STATUS_POINT, STATUS_RADIUS
export STATUS_EDGE, STATUS_SHEET, STATUS_NONFINITE, STATUS_INPUT

const QMAX = 6
const NM = 84
const WORKPREC = 256

const STATUS_OK = 0
const STATUS_ORDER = 5
const STATUS_POINT = 6
const STATUS_RADIUS = 7
const STATUS_EDGE = 8
const STATUS_SHEET = 9
const STATUS_NONFINITE = 10
const STATUS_INPUT = 11

const EXPS6 = let
    v = NTuple{3,Int}[]
    sizehint!(v, NM)
    for deg in 0:QMAX
        for a in 0:deg
            for b in 0:(deg - a)
                push!(v, (a, b, deg - a - b))
            end
        end
    end
    @assert length(v) == NM
    Tuple(v)
end

const IDX6 = let
    a = zeros(Int, QMAX + 1, QMAX + 1, QMAX + 1)
    for (i, (x, y, z)) in enumerate(EXPS6)
        a[x + 1, y + 1, z + 1] = i
    end
    a
end

const NMONO = ntuple(q -> (q * (q + 1) * (q + 2)) ÷ 6, QMAX + 1)

nmono(order::Integer) = (0 <= order <= QMAX) ? NMONO[order + 1] : 0

exponents(order::Integer) = (0 <= order <= QMAX) ? collect(EXPS6[1:nmono(order)]) :
                            NTuple{3,Int}[]

struct FieldResult
    status::Int
    exponents::Vector{NTuple{3,Int}}
    coefficients::Matrix{Float64}
end

function empty_result(status::Int)
    return FieldResult(status, NTuple{3,Int}[], Array{Float64}(undef, 3, 0))
end

function ok_result(order::Int, coeff::Matrix{Float64})
    return FieldResult(STATUS_OK, exponents(order), coeff)
end

# ---------------------------------------------------------------------------
# Exact binary64 rational membership
# ---------------------------------------------------------------------------

function exact_rational(x::Float64)
    isfinite(x) || throw(ArgumentError("nonfinite binary64"))
    r = Rational{BigInt}(x)
    Float64(r) == x || throw(ArgumentError("inexact binary64 rational conversion"))
    return r
end

struct RVec
    x::Rational{BigInt}
    y::Rational{BigInt}
    z::Rational{BigInt}
end

RVec(p::NTuple{3,Float64}) = RVec(exact_rational(p[1]), exact_rational(p[2]), exact_rational(p[3]))

Base.:-(a::RVec, b::RVec) = RVec(a.x - b.x, a.y - b.y, a.z - b.z)
Base.:+(a::RVec, b::RVec) = RVec(a.x + b.x, a.y + b.y, a.z + b.z)
Base.:(==)(a::RVec, b::RVec) = a.x == b.x && a.y == b.y && a.z == b.z
dot(a::RVec, b::RVec) = a.x * b.x + a.y * b.y + a.z * b.z
function cross(a::RVec, b::RVec)
    return RVec(a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x)
end
iszero_vec(a::RVec) = iszero(a.x) && iszero(a.y) && iszero(a.z)

@enum PointClass begin
    CLASS_OFFPLANE
    CLASS_EXTERIOR_PLANE
    CLASS_INTERIOR
    CLASS_EDGE
    CLASS_VERTEX
    CLASS_DEGENERATE
end

function on_segment(p::RVec, a::RVec, b::RVec)
    d = b - a
    w = p - a
    iszero_vec(cross(w, d)) || return false
    t = dot(w, d)
    len2 = dot(d, d)
    return t > 0 && t < len2
end

function classify_point(A::RVec, B::RVec, C::RVec, P::RVec)
    N = cross(B - A, C - A)
    iszero_vec(N) && return CLASS_DEGENERATE
    (P == A || P == B || P == C) && return CLASS_VERTEX
    on_segment(P, A, B) && return CLASS_EDGE
    on_segment(P, B, C) && return CLASS_EDGE
    on_segment(P, C, A) && return CLASS_EDGE
    on_plane = iszero(dot(P - A, N))
    on_plane || return CLASS_OFFPLANE
    s1 = dot(cross(B - A, P - A), N)
    s2 = dot(cross(C - B, P - B), N)
    s3 = dot(cross(A - C, P - C), N)
    if s1 > 0 && s2 > 0 && s3 > 0
        return CLASS_INTERIOR
    end
    return CLASS_EXTERIOR_PLANE
end

function classify_triangle(A::NTuple{3,Float64}, B::NTuple{3,Float64},
                           C::NTuple{3,Float64}, P::NTuple{3,Float64})
    return classify_point(RVec(A), RVec(B), RVec(C), RVec(P))
end

# ---------------------------------------------------------------------------
# Real multivariate Taylor jets (scoped BigFloat)
# ---------------------------------------------------------------------------

struct Jet
    order::Int
    c::Vector{BigFloat}
end

function jet_zero(order::Int)
    n = nmono(order)
    return Jet(order, BigFloat[zero(BigFloat) for _ in 1:n])
end

function jet_const(val::BigFloat, order::Int)
    j = jet_zero(order)
    j.c[1] = copy(val)
    return j
end

function jet_coord(center::BigFloat, axis::Int, radius::BigFloat, order::Int)
    j = jet_const(center, order)
    if order > 0
        a = axis == 1 ? 1 : 0
        b = axis == 2 ? 1 : 0
        c = axis == 3 ? 1 : 0
        j.c[IDX6[a + 1, b + 1, c + 1]] = copy(radius)
    end
    return j
end

function axpy!(s::BigFloat, a::Jet, b::Jet)
    n = nmono(a.order)
    @inbounds for i in 1:n
        b.c[i] += s * a.c[i]
    end
    return b
end

function add(a::Jet, b::Jet)
    n = nmono(a.order)
    c = jet_zero(a.order)
    @inbounds for i in 1:n
        c.c[i] = a.c[i] + b.c[i]
    end
    return c
end

function sub(a::Jet, b::Jet)
    n = nmono(a.order)
    c = jet_zero(a.order)
    @inbounds for i in 1:n
        c.c[i] = a.c[i] - b.c[i]
    end
    return c
end

function neg(a::Jet)
    n = nmono(a.order)
    c = jet_zero(a.order)
    @inbounds for i in 1:n
        c.c[i] = -a.c[i]
    end
    return c
end

function scale(a::Jet, s::BigFloat)
    n = nmono(a.order)
    c = jet_zero(a.order)
    @inbounds for i in 1:n
        c.c[i] = s * a.c[i]
    end
    return c
end

function mul(a::Jet, b::Jet)
    q = a.order
    n = nmono(q)
    c = jet_zero(q)
    @inbounds for i in 1:n
        ai = a.c[i]
        iszero(ai) && continue
        ax, ay, az = EXPS6[i]
        for j in 1:n
            bj = b.c[j]
            iszero(bj) && continue
            bx, by, bz = EXPS6[j]
            sx, sy, sz = ax + bx, ay + by, az + bz
            sx + sy + sz > q && continue
            k = IDX6[sx + 1, sy + 1, sz + 1]
            c.c[k] += ai * bj
        end
    end
    return c
end

function pow(a::Jet, exponent::BigFloat)
    q = a.order
    a0 = a.c[1]
    iszero(a0) && return nothing
    if iszero(exponent)
        return jet_const(one(BigFloat), q)
    end
    if exponent == 1
        return Jet(q, copy.(a.c))
    end
    u = scale(sub(a, jet_const(a0, q)), inv(a0))
    total = jet_const(one(BigFloat), q)
    term = jet_const(one(BigFloat), q)
    coeff = one(BigFloat)
    for k in 1:q
        term = mul(term, u)
        coeff *= (exponent - BigFloat(k - 1)) / BigFloat(k)
        axpy!(coeff, term, total)
    end
    return scale(total, a0^exponent)
end

function inv_jet(a::Jet)
    return pow(a, -one(BigFloat))
end

function sqrt_jet(a::Jet)
    a.c[1] < 0 && return nothing
    return pow(a, BigFloat(1) / BigFloat(2))
end

function log_jet(a::Jet)
    q = a.order
    a0 = a.c[1]
    a0 <= 0 && return nothing
    u = scale(sub(a, jet_const(a0, q)), inv(a0))
    total = jet_const(log(a0), q)
    term = jet_const(one(BigFloat), q)
    s = one(BigFloat)
    for k in 1:q
        term = mul(term, u)
        s = -s
        axpy!(-s / BigFloat(k), term, total)
    end
    return total
end

# atan(t) about t0 = 0 using odd powers: t - t^3/3 + t^5/5 - ...
function atan_odd(t::Jet)
    q = t.order
    result = jet_zero(q)
    tk = Jet(q, copy.(t.c))
    sgn = one(BigFloat)
    k = 1
    while k <= q
        axpy!(sgn / BigFloat(k), tk, result)
        k + 2 > q && break
        tk = mul(mul(tk, t), t)
        k += 2
        sgn = -sgn
    end
    return result
end

# atan2(y, x) = atan2(y0, x0) + atan(t), t = (x0 y - y0 x)/(x0 x + y0 y), t0 = 0
function atan2_jet(y::Jet, x::Jet)
    x0 = x.c[1]
    y0 = y.c[1]
    r2 = x0 * x0 + y0 * y0
    iszero(r2) && return nothing
    num = sub(scale(y, x0), scale(x, y0))
    den = add(scale(x, x0), scale(y, y0))
    deni = inv_jet(den)
    deni === nothing && return nothing
    t = mul(num, deni)
    t.c[1] = zero(BigFloat)
    return add(jet_const(atan(y0, x0), x.order), atan_odd(t))
end

function jet_finite(a::Jet)
    n = nmono(a.order)
    @inbounds for i in 1:n
        isfinite(a.c[i]) || return false
    end
    return true
end

struct Jet3
    x::Jet
    y::Jet
    z::Jet
end

function jet3_point(center::NTuple{3,BigFloat}, radius::BigFloat, order::Int)
    return Jet3(jet_coord(center[1], 1, radius, order),
                jet_coord(center[2], 2, radius, order),
                jet_coord(center[3], 3, radius, order))
end

function jet3_const(v::NTuple{3,BigFloat}, order::Int)
    return Jet3(jet_const(v[1], order), jet_const(v[2], order), jet_const(v[3], order))
end

sub3(a::Jet3, b::Jet3) = Jet3(sub(a.x, b.x), sub(a.y, b.y), sub(a.z, b.z))
scale3(a::Jet3, s::BigFloat) = Jet3(scale(a.x, s), scale(a.y, s), scale(a.z, s))
dot3(a::Jet3, b::Jet3) = add(add(mul(a.x, b.x), mul(a.y, b.y)), mul(a.z, b.z))
function cross3(a::Jet3, b::Jet3)
    return Jet3(sub(mul(a.y, b.z), mul(a.z, b.y)),
                sub(mul(a.z, b.x), mul(a.x, b.z)),
                sub(mul(a.x, b.y), mul(a.y, b.x)))
end
norm2_3(a::Jet3) = dot3(a, a)
function norm3(a::Jet3)
    s = sqrt_jet(norm2_3(a))
    return s
end
jet3_finite(a::Jet3) = jet_finite(a.x) && jet_finite(a.y) && jet_finite(a.z)

# ---------------------------------------------------------------------------
# Closed-form field / Taylor map
# ---------------------------------------------------------------------------

bf3(p::NTuple{3,Float64}) = (BigFloat(p[1]), BigFloat(p[2]), BigFloat(p[3]))

function f64_finite3(p::NTuple{3,Float64})
    return isfinite(p[1]) && isfinite(p[2]) && isfinite(p[3])
end

function integer_order(order_f::Float64)
    isfinite(order_f) || return nothing
    order_f == trunc(order_f) || return nothing
    (0 <= order_f <= QMAX) || return nothing
    return Int(order_f)
end

function to_f64_coeff(bx::Jet, by::Jet, bz::Jet)
    n = nmono(bx.order)
    coeff = Matrix{Float64}(undef, 3, n)
    @inbounds for k in 1:n
        x = Float64(bx.c[k])
        y = Float64(by.c[k])
        z = Float64(bz.c[k])
        (isfinite(x) && isfinite(y) && isfinite(z)) || return nothing
        coeff[1, k] = x
        coeff[2, k] = y
        coeff[3, k] = z
    end
    return coeff
end

function eval_kernel(order::Int, radius::Float64, sigma::Float64,
                     A::NTuple{3,Float64}, B::NTuple{3,Float64},
                     C::NTuple{3,Float64}, P::NTuple{3,Float64},
                     class::PointClass)
    return setprecision(BigFloat, WORKPREC) do
        eval_kernel_prec(order, radius, sigma, A, B, C, P, class)
    end
end

function eval_kernel_prec(order::Int, radius::Float64, sigma::Float64,
                          A::NTuple{3,Float64}, B::NTuple{3,Float64},
                          C::NTuple{3,Float64}, P::NTuple{3,Float64},
                          class::PointClass)
    a = bf3(A)
    b = bf3(B)
    c = bf3(C)
    p0 = bf3(P)
    eab = (b[1] - a[1], b[2] - a[2], b[3] - a[3])
    eac = (c[1] - a[1], c[2] - a[2], c[3] - a[3])
    Nx = eab[2] * eac[3] - eab[3] * eac[2]
    Ny = eab[3] * eac[1] - eab[1] * eac[3]
    Nz = eab[1] * eac[2] - eab[2] * eac[1]
    nmag = sqrt(Nx * Nx + Ny * Ny + Nz * Nz)
    iszero(nmag) && return empty_result(STATUS_INPUT)
    lscale = max(sqrt(eab[1]^2 + eab[2]^2 + eab[3]^2),
                 sqrt((c[1] - b[1])^2 + (c[2] - b[2])^2 + (c[3] - b[3])^2),
                 sqrt((a[1] - c[1])^2 + (a[2] - c[2])^2 + (a[3] - c[3])^2))
    iszero(lscale) && return empty_result(STATUS_INPUT)
    invs = inv(lscale)
    vs = (a, b, c)
    vsc = ntuple(i -> ((vs[i][1] - a[1]) * invs,
                       (vs[i][2] - a[2]) * invs,
                       (vs[i][3] - a[3]) * invs), 3)
    psc = ((p0[1] - a[1]) * invs, (p0[2] - a[2]) * invs, (p0[3] - a[3]) * invs)
    rad = BigFloat(radius) * invs
    Ns = (Nx * invs * invs, Ny * invs * invs, Nz * invs * invs)
    nsmag = nmag * invs * invs
    nh = (Ns[1] / nsmag, Ns[2] / nsmag, Ns[3] / nsmag)
    pj = jet3_point(psc, rad, order)
    Aj = jet3_const(vsc[1], order)
    Nraw = jet3_const(Ns, order)
    # Conditioned numerator from the fixed unnormalized normal.
    num = neg(dot3(sub3(pj, Aj), Nraw))
    if class == CLASS_EXTERIOR_PLANE || class == CLASS_INTERIOR
        # Exact plane: zero the known-zero constant; keep higher coefficients.
        num.c[1] = zero(BigFloat)
    end
    r = ntuple(i -> sub3(jet3_const(vsc[i], order), pj), 3)
    rho = Jet[]
    sizehint!(rho, 3)
    for i in 1:3
        ri = norm3(r[i])
        ri === nothing && return empty_result(STATUS_EDGE)
        iszero(ri.c[1]) && return empty_result(STATUS_EDGE)
        push!(rho, ri)
    end
    den = mul(mul(rho[1], rho[2]), rho[3])
    for i in 1:3
        j = i == 3 ? 1 : i + 1
        k = j == 3 ? 1 : j + 1
        den = add(den, mul(dot3(r[i], r[j]), rho[k]))
    end
    omega = jet_zero(order)
    if class == CLASS_INTERIOR && order == 0
        omega = jet_zero(order)
    else
        om = atan2_jet(num, den)
        om === nothing && return empty_result(STATUS_INPUT)
        omega = scale(om, BigFloat(2))
    end
    Kx = scale(omega, -nh[1])
    Ky = scale(omega, -nh[2])
    Kz = scale(omega, -nh[3])
    verts = vsc
    for i in 1:3
        j = i == 3 ? 1 : i + 1
        chord = (verts[j][1] - verts[i][1],
                 verts[j][2] - verts[i][2],
                 verts[j][3] - verts[i][3])
        length = sqrt(chord[1]^2 + chord[2]^2 + chord[3]^2)
        iszero(length) && return empty_result(STATUS_INPUT)
        t = (chord[1] / length, chord[2] / length, chord[3] / length)
        tj = jet3_const(t, order)
        chj = jet3_const(chord, order)
        vi = jet3_const(verts[i], order)
        vj = jet3_const(verts[j], order)
        u = dot3(sub3(pj, vi), tj)
        w = dot3(sub3(vj, pj), tj)
        u0 = u.c[1]
        w0 = w.c[1]
        su = u0 < 0 ? -1 : 1
        sw = w0 < 0 ? -1 : 1
        offset = abs(u0) <= abs(w0) ? sub3(pj, vi) : sub3(pj, vj)
        cr = cross3(offset, chj)
        rho2 = scale(norm2_3(cr), inv(length * length))
        tmp1 = add(rho[i], scale(u, BigFloat(su)))
        tmp2 = add(rho[j], scale(w, BigFloat(sw)))
        (iszero(tmp1.c[1]) || iszero(tmp2.c[1])) && return empty_result(STATUS_EDGE)
        i1 = inv_jet(tmp1)
        i2 = inv_jet(tmp2)
        (i1 === nothing || i2 === nothing) && return empty_result(STATUS_INPUT)
        gap = add(scale(u, BigFloat(su - 1)), scale(w, BigFloat(sw - 1)))
        gap = add(gap, mul(rho2, add(i1, i2)))
        gap.c[1] <= 0 && return empty_result(STATUS_INPUT)
        ig = inv_jet(gap)
        ig === nothing && return empty_result(STATUS_INPUT)
        arg = add(jet_const(one(BigFloat), order), scale(ig, 2 * length))
        L = log_jet(arg)
        L === nothing && return empty_result(STATUS_INPUT)
        # outward = t × n
        ox = t[2] * nh[3] - t[3] * nh[2]
        oy = t[3] * nh[1] - t[1] * nh[3]
        oz = t[1] * nh[2] - t[2] * nh[1]
        axpy!(ox, L, Kx)
        axpy!(oy, L, Ky)
        axpy!(oz, L, Kz)
    end
    (jet_finite(Kx) && jet_finite(Ky) && jet_finite(Kz)) || return empty_result(STATUS_NONFINITE)
    fac = BigFloat(sigma) / (BigFloat(4) * BigFloat(π))
    Bx = scale(Kx, fac)
    By = scale(Ky, fac)
    Bz = scale(Kz, fac)
    (jet_finite(Bx) && jet_finite(By) && jet_finite(Bz)) || return empty_result(STATUS_NONFINITE)
    coeff = to_f64_coeff(Bx, By, Bz)
    coeff === nothing && return empty_result(STATUS_NONFINITE)
    return ok_result(order, coeff)
end

function zero_coeff(order::Int)
    n = nmono(order)
    return ok_result(order, zeros(Float64, 3, n))
end

"""
    triangle_map(order, radius, sigma, A..., P...) -> FieldResult

Evaluate the radius-scaled Taylor map of μ0 H. Inputs are binary64. Order,
radius and the observer are checked before a zero-charge short-circuit.
"""
function triangle_map(order_f::Float64, radius::Float64, sigma::Float64,
                     ax::Float64, ay::Float64, az::Float64,
                     bx::Float64, by::Float64, bz::Float64,
                     cx::Float64, cy::Float64, cz::Float64,
                     px::Float64, py::Float64, pz::Float64)
    q = integer_order(order_f)
    q === nothing && return empty_result(STATUS_ORDER)
    (isfinite(radius) && radius > 0) || return empty_result(STATUS_RADIUS)
    P = (px, py, pz)
    f64_finite3(P) || return empty_result(STATUS_POINT)
    A = (ax, ay, az)
    B = (bx, by, bz)
    C = (cx, cy, cz)
    if !(f64_finite3(A) && f64_finite3(B) && f64_finite3(C) && isfinite(sigma))
        return empty_result(STATUS_INPUT)
    end
    class = try
        classify_triangle(A, B, C, P)
    catch
        return empty_result(STATUS_INPUT)
    end
    class == CLASS_DEGENERATE && return empty_result(STATUS_INPUT)
    if iszero(sigma)
        return zero_coeff(q)
    end
    if class == CLASS_EDGE || class == CLASS_VERTEX
        return empty_result(STATUS_EDGE)
    end
    if class == CLASS_INTERIOR && q > 0
        return empty_result(STATUS_SHEET)
    end
    return eval_kernel(q, radius, sigma, A, B, C, P, class)
end

function triangle_map(vals::AbstractVector{<:Real})
    length(vals) == 15 || return empty_result(STATUS_INPUT)
    v = ntuple(i -> Float64(vals[i]), 15)
    return triangle_map(v...)
end

function evaluate_map(coeff::AbstractMatrix, exps, u::NTuple{3,Float64})
    size(coeff) == (3, length(exps)) || throw(DimensionMismatch("coefficients must have shape (3, length(exponents))"))
    Base.require_one_based_indexing(coeff, exps)
    s1 = 0.0
    s2 = 0.0
    s3 = 0.0
    for k in eachindex(exps)
        a, b, c = exps[k]
        m = u[1]^a * u[2]^b * u[3]^c
        s1 += coeff[1, k] * m
        s2 += coeff[2, k] * m
        s3 += coeff[3, k] * m
    end
    return (s1, s2, s3)
end

# ---------------------------------------------------------------------------
# JSON (no JSON runtime package)
# ---------------------------------------------------------------------------

function json_number(x::Float64)
    isfinite(x) || return "0"
    return string(x)
end

function write_result_json(io::IO, r::FieldResult)
    print(io, "{\"status\":", r.status, ",\"exponents\":[")
    for i in eachindex(r.exponents)
        i > 1 && print(io, ',')
        a, b, c = r.exponents[i]
        print(io, '[', a, ',', b, ',', c, ']')
    end
    print(io, "],\"coefficients\":")
    if r.status != STATUS_OK || isempty(r.exponents)
        print(io, "[]")
    else
        print(io, '[')
        n = size(r.coefficients, 2)
        for row in 1:3
            row > 1 && print(io, ',')
            print(io, '[')
            for k in 1:n
                k > 1 && print(io, ',')
                print(io, json_number(r.coefficients[row, k]))
            end
            print(io, ']')
        end
        print(io, ']')
    end
    print(io, '}')
    return
end

function result_json(r::FieldResult)
    buf = IOBuffer()
    write_result_json(buf, r)
    return String(take!(buf))
end

function eval_standalone_tokens(tokens::Vector{String})
    if length(tokens) != 15
        return empty_result(STATUS_INPUT)
    end
    vals = Vector{Float64}(undef, 15)
    for i in 1:15
        v = tryparse(Float64, tokens[i])
        v === nothing && return empty_result(STATUS_INPUT)
        vals[i] = v
    end
    return triangle_map(vals)
end

# ---------------------------------------------------------------------------
# Native Fortran client (shared tan_mod physical kernel, C ABI)
# ---------------------------------------------------------------------------

mutable struct NativeState
    libpath::String
    handle::Ptr{Cvoid}
    cache::String
end

const NATIVE = NativeState("", C_NULL, "")

function native_close!()
    if NATIVE.handle != C_NULL
        try
            # Loading an empty path clears native allocations before unloading.
            status = Ref{Cint}(0)
            ccall(Libdl.dlsym(NATIVE.handle, :tan_load_c), Cvoid,
                  (Cstring, Ptr{Cint}), "", status)
            Libdl.dlclose(NATIVE.handle)
        catch
        end
    end
    NATIVE.libpath = ""
    NATIVE.handle = C_NULL
    NATIVE.cache = ""
    return
end

function native_open!(libpath::AbstractString)
    if occursin('\0', libpath)
        native_close!()
        return STATUS_INPUT
    end
    ap = abspath(libpath)
    if NATIVE.handle != C_NULL && NATIVE.libpath == ap
        return STATUS_OK
    end
    native_close!()
    isfile(ap) || return 2
    h = try
        Libdl.dlopen(ap, Libdl.RTLD_LAZY | Libdl.RTLD_GLOBAL)
    catch
        C_NULL
    end
    h == C_NULL && return 2
    if any(Libdl.dlsym_e(h, symbol) == C_NULL for symbol in
           (:tan_load_c, :tan_eval_c, :tan_last_mode_c))
        Libdl.dlclose(h)
        return STATUS_INPUT
    end
    NATIVE.handle = h
    NATIVE.libpath = ap
    NATIVE.cache = ""
    return STATUS_OK
end

function native_load!(libpath::AbstractString, cache::AbstractString)
    st = native_open!(libpath)
    st == STATUS_OK || return st
    status = Ref{Cint}(0)
    if occursin('\0', cache)
        ccall(Libdl.dlsym(NATIVE.handle, :tan_load_c), Cvoid,
              (Cstring, Ptr{Cint}), "", status)
        NATIVE.cache = ""
        return STATUS_INPUT
    end
    ccall(Libdl.dlsym(NATIVE.handle, :tan_load_c), Cvoid,
          (Cstring, Ptr{Cint}), String(cache), status)
    if status[] == 0
        NATIVE.cache = abspath(cache)
    else
        NATIVE.cache = ""
    end
    return Int(status[])
end

function native_last_mode()
    NATIVE.handle == C_NULL && return -1
    return Int(ccall(Libdl.dlsym(NATIVE.handle, :tan_last_mode_c), Cint, ()))
end

"""
    native_eval(lib, cache, x, y, z, mode, order, radius) -> FieldResult

Call tan_load_c / tan_eval_c. This shares the Fortran physical kernel.
The Fortran cache is process-global; do not use this from multiple threads.
"""
function native_eval(libpath::AbstractString, cache::AbstractString,
                     x::Float64, y::Float64, z::Float64,
                     mode::Int, order::Int, radius::Float64)
    (isfinite(x) && isfinite(y) && isfinite(z)) || return empty_result(STATUS_POINT)
    isfinite(radius) || return empty_result(STATUS_RADIUS)
    radius > 0 || return empty_result(STATUS_RADIUS)
    (0 <= order <= QMAX) || return empty_result(STATUS_ORDER)
    (mode == 0 || mode == 1 || mode == 2) || return empty_result(4)
    if (mode == 0 || mode == 2) && order != 0
        return empty_result(4)
    end
    st = native_load!(libpath, cache)
    st == 0 || return empty_result(st)
    point = [x, y, z]
    coeff = zeros(Float64, 3, NM)
    status = Ref{Cint}(0)
    ccall(Libdl.dlsym(NATIVE.handle, :tan_eval_c), Cvoid,
          (Ptr{Float64}, Cint, Cint, Float64, Ptr{Float64}, Ptr{Cint}),
          point, Cint(mode), Cint(order), radius, coeff, status)
    s = Int(status[])
    s == 0 || return empty_result(s)
    n = nmono(order)
    out = Matrix{Float64}(undef, 3, n)
    @inbounds for k in 1:n
        for row in 1:3
            v = coeff[row, k]
            isfinite(v) || return empty_result(STATUS_NONFINITE)
            out[row, k] = v
        end
    end
    return ok_result(order, out)
end

function eval_native_tokens(tokens::Vector{String})
    if length(tokens) != 8
        return empty_result(STATUS_INPUT)
    end
    lib = tokens[1]
    cache = tokens[2]
    nums = Vector{Float64}(undef, 6)
    for i in 1:6
        v = tryparse(Float64, tokens[i + 2])
        v === nothing && return empty_result(STATUS_INPUT)
        nums[i] = v
    end
    mode_f, order_f = nums[4], nums[5]
    mode_f == trunc(mode_f) && isfinite(mode_f) || return empty_result(4)
    order_f == trunc(order_f) && isfinite(order_f) || return empty_result(STATUS_ORDER)
    (0 <= mode_f <= 2) || return empty_result(4)
    (0 <= order_f <= QMAX) || return empty_result(STATUS_ORDER)
    return native_eval(lib, cache, nums[1], nums[2], nums[3],
                       Int(mode_f), Int(order_f), nums[6])
end

end # module
