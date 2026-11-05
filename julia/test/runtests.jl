# Independent tests for the standalone triangle map. Quadrature and closed
# spherical-excess controls; no parent-port arrays as oracle.
include(joinpath(@__DIR__, "..", "src", "TriangularPMFields.jl"))
using .TriangularPMFields
using Test

const PKG = joinpath(@__DIR__, "..")
const JULIA = Base.julia_cmd()
const WORKPREC_OUTSIDE = precision(BigFloat)

relmax(a, b) = maximum(abs.(a .- b)) / max(1.0, maximum(abs.(b)))
nvec(x) = sqrt(x[1]^2 + x[2]^2 + x[3]^2)
cross3(a, b) = (a[2]*b[3]-a[3]*b[2], a[3]*b[1]-a[1]*b[3], a[1]*b[2]-a[2]*b[1])
dot3(a, b) = a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
sub3(a, b) = (a[1]-b[1], a[2]-b[2], a[3]-b[3])
add3(a, b) = (a[1]+b[1], a[2]+b[2], a[3]+b[3])
scale3(a, s) = (a[1]*s, a[2]*s, a[3]*s)

function parse_json_status(s::AbstractString)
    m = match(r"\"status\":(-?\d+)", s)
    m === nothing && return nothing
    return parse(Int, m.captures[1])
end

function gauss_legendre(n::Int)
    x = Vector{Float64}(undef, n)
    w = Vector{Float64}(undef, n)
    for i in 1:n
        z = cos(π * (i - 0.25) / (n + 0.5))
        p = 0.0
        dp = 0.0
        for _ in 1:12
            p0 = 1.0
            p1 = z
            for k in 1:(n - 1)
                p0, p1 = p1, (((2k + 1) * z * p1) - k * p0) / (k + 1)
            end
            p = p1
            dp = n * (z * p1 - p0) / (z * z - 1)
            z -= p / dp
        end
        x[i] = z
        w[i] = 2 / ((1 - z * z) * dp * dp)
    end
    return x, w
end

function triangle_nodes(A, B, C, order::Int)
    t, w = gauss_legendre(order)
    t = (t .+ 1) ./ 2
    w = w ./ 2
    AB = sub3(B, A)
    AC = sub3(C, A)
    area2 = nvec(cross3(AB, AC))
    nodes = NTuple{3,Float64}[]
    weights = Float64[]
    for i in 1:order
        u = t[i]
        wu = w[i]
        for j in 1:order
            v = t[j]
            wv = w[j]
            # A + u AB + (1-u) v AC
            p = add3(A, add3(scale3(AB, u), scale3(AC, (1 - u) * v)))
            push!(nodes, p)
            push!(weights, wu * wv * (1 - u) * area2)
        end
    end
    return nodes, weights
end

function quadrature_K(A, B, C, P, order::Int)
    nodes, weights = triangle_nodes(A, B, C, order)
    s = (0.0, 0.0, 0.0)
    for (q, wt) in zip(nodes, weights)
        d = sub3(P, q)
        r = nvec(d)
        s = add3(s, scale3(d, wt / (r * r * r)))
    end
    return s
end

function spherical_excess_solid_angle(A, B, C, P)
    ra = sub3(A, P)
    rb = sub3(B, P)
    rc = sub3(C, P)
    na = nvec(ra); nb = nvec(rb); nc = nvec(rc)
    ua = scale3(ra, 1 / na)
    ub = scale3(rb, 1 / nb)
    uc = scale3(rc, 1 / nc)
    function ang(u, v, w)
        # spherical angle at u between v and w
        n1 = cross3(u, v)
        n2 = cross3(u, w)
        s1 = nvec(n1); s2 = nvec(n2)
        c = dot3(n1, n2) / (s1 * s2)
        c = clamp(c, -1.0, 1.0)
        s = dot3(cross3(n1, n2), u) / (s1 * s2)
        return atan(s, c)
    end
    α = ang(ua, ub, uc)
    β = ang(ub, uc, ua)
    γ = ang(uc, ua, ub)
    Ω = α + β + γ - π
    # Oriented excess can land 2π away from the principal solid angle.
    while Ω > π
        Ω -= 2π
    end
    while Ω < -π
        Ω += 2π
    end
    return Ω
end

function quadrature_jet_K(A, B, C, P, order::Int, qorder::Int, radius::Float64)
    keys = exponents(order)
    n = length(keys)
    lookup = Dict{NTuple{3,Int},Int}()
    for (i, k) in enumerate(keys)
        lookup[k] = i
    end
    nodes, weights = triangle_nodes(A, B, C, qorder)
    result = zeros(Float64, 3, n)
    qterms = Tuple{NTuple{3,Int},Vector{Float64}}[]
    for axis in 1:3
        key1 = ntuple(i -> Int(i == axis), 3)
        key2 = ntuple(i -> 2 * Int(i == axis), 3)
        # filled per node below
        push!(qterms, (key1, zeros(length(nodes))))
        push!(qterms, (key2, zeros(length(nodes))))
    end
    dmat = [sub3(P, q) for q in nodes]
    r2 = [dot3(d, d) for d in dmat]
    for (iq, d) in enumerate(dmat)
        qterms[1][2][iq] = 2 * d[1] / r2[iq]
        qterms[2][2][iq] = 1 / r2[iq]
        qterms[3][2][iq] = 2 * d[2] / r2[iq]
        qterms[4][2][iq] = 1 / r2[iq]
        qterms[5][2][iq] = 2 * d[3] / r2[iq]
        qterms[6][2][iq] = 1 / r2[iq]
    end
    power = zeros(Float64, n, length(nodes))
    power[1, :] .= 1
    series = copy(power)
    coefficient = 1.0
    for k in 1:order
        newp = zeros(Float64, n, length(nodes))
        for i in 1:n
            a = keys[i]
            for (b, values) in qterms
                target = (a[1] + b[1], a[2] + b[2], a[3] + b[3])
                if target[1] + target[2] + target[3] <= order
                    newp[lookup[target], :] .+= power[i, :] .* values
                end
            end
        end
        power = newp
        coefficient *= (-1.5 - k + 1) / k
        series .+= coefficient .* power
    end
    base_w = [weights[i] / r2[i]^1.5 for i in eachindex(nodes)]
    for axis in 1:3
        c = similar(series)
        for i in 1:n
            for j in eachindex(nodes)
                c[i, j] = series[i, j] * dmat[j][axis]
            end
        end
        for (i, key) in enumerate(keys)
            if key[axis] > 0
                prev = ntuple(t -> t == axis ? key[t] - 1 : key[t], 3)
                c[i, :] .+= series[lookup[prev], :]
            end
        end
        for i in 1:n
            acc = 0.0
            for j in eachindex(nodes)
                acc += c[i, j] * base_w[j]
            end
            result[axis, i] = acc * radius^(sum(keys[i]))
        end
    end
    return result
end

function mu0H_from_K(K, sigma)
    return scale3(K, sigma / (4 * π))
end

function maxwell_residual(coeff, order)
    keys = exponents(order)
    lookup = Dict(k => i for (i, k) in enumerate(keys))
    divm = 0.0
    curlm = 0.0
    nrm = max(1.0, maximum(abs.(coeff)))
    for k in exponents(order - 1)
        deriv = zeros(3, 3)
        for axis in 1:3
            higher = ntuple(t -> t == axis ? k[t] + 1 : k[t], 3)
            deriv[:, axis] .= (k[axis] + 1) .* coeff[:, lookup[higher]]
        end
        divm = max(divm, abs(deriv[1, 1] + deriv[2, 2] + deriv[3, 3]))
        curlm = max(curlm, abs(deriv[3, 2] - deriv[2, 3]),
                    abs(deriv[1, 3] - deriv[3, 1]),
                    abs(deriv[2, 1] - deriv[1, 2]))
    end
    return divm / nrm, curlm / nrm
end

function run_cli(args::Vector{String})
    iob = IOBuffer()
    proc = run(pipeline(ignorestatus(`$(JULIA) --startup-file=no $(joinpath(PKG, "cli.jl")) $args`), iob))
    return String(take!(iob)), proc.exitcode == 0
end

function run_cli_batch(body::String)
    iob = IOBuffer()
    run(pipeline(ignorestatus(`$(JULIA) --startup-file=no $(joinpath(PKG, "cli.jl")) --batch`);
                 stdin=IOBuffer(body), stdout=iob))
    return String(take!(iob))
end

const A0 = (0.0, 0.0, 0.0)
const B0 = (2.0, 0.0, 0.0)
const C0 = (0.0, 2.0, 0.0)
const SIG = 0.8
const PREC = 1e-10

@testset "TriangularPMFields" begin
    @testset "exact rational conversion" begin
        samples = Float64[0.0, -0.0, 1.0, -1.0, 0.1, 1.2, 0.2, 0.9,
                          floatmin(Float64), nextfloat(0.0), eps(Float64),
                          1.0e-300, π, 1.2e10, -7.5, 0.5]
        for x in samples
            r = TriangularPMFields.exact_rational(x)
            @test Float64(r) == x
        end
        r01 = TriangularPMFields.exact_rational(0.1)
        @test r01 == 3602879701896397 // 36028797018963968
        @test Float64(r01) === 0.1
    end

    @testset "exponent order and nmono" begin
        @test nmono.((0, 1, 2, 3, 4, 5, 6)) == (1, 4, 10, 20, 35, 56, 84)
        @test nmono(7) == 0
        e = exponents(6)
        @test length(e) == 84
        @test e[1] == (0, 0, 0)
        @test e[2] == (0, 0, 1)
        @test e[3] == (0, 1, 0)
        @test e[4] == (1, 0, 0)
        @test e[10] == (2, 0, 0)
        @test e[84] == (6, 0, 0)
        for deg in 0:6
            block = [t for t in e if sum(t) == deg]
            @test issorted(block)
        end
        @test precision(BigFloat) == WORKPREC_OUTSIDE
    end

    @testset "scalar quadrature and spherical excess" begin
        P = (0.4, 0.3, 0.5)
        r = triangle_map(0.0, 1.0, SIG, A0..., B0..., C0..., P...)
        @test r.status == STATUS_OK
        @test size(r.coefficients) == (3, 1)
        q32 = mu0H_from_K(quadrature_K(A0, B0, C0, P, 32), SIG)
        q48 = mu0H_from_K(quadrature_K(A0, B0, C0, P, 48), SIG)
        got = (r.coefficients[1, 1], r.coefficients[2, 1], r.coefficients[3, 1])
        @test nvec(sub3(q32, q48)) / max(1.0, nvec(q48)) <= 1e-11
        @test nvec(sub3(got, q48)) <= 1e-12 + 1e-10 * nvec(q48)
        N = cross3(sub3(B0, A0), sub3(C0, A0))
        nh = scale3(N, 1 / nvec(N))
        Ω = spherical_excess_solid_angle(A0, B0, C0, P)
        @test abs(dot3(got, nh) + SIG * Ω / (4 * π)) < 1e-12
        @test precision(BigFloat) == WORKPREC_OUTSIDE
    end

    @testset "zero charge after validation" begin
        Pv = A0
        r = triangle_map(6.0, 1.0, 0.0, A0..., B0..., C0..., Pv...)
        @test r.status == STATUS_OK
        @test size(r.coefficients) == (3, 84)
        @test all(iszero, r.coefficients)
        r2 = triangle_map(0.0, 1.0, 0.0, A0..., B0..., C0..., (0.5, 0.5, 0.0)...)
        @test r2.status == STATUS_OK
        @test all(iszero, r2.coefficients)
        bad = triangle_map(7.0, 1.0, 0.0, A0..., B0..., C0..., Pv...)
        @test bad.status == STATUS_ORDER
        badp = triangle_map(0.0, 1.0, 0.0, A0..., B0..., C0..., NaN, 0.0, 0.0)
        @test badp.status == STATUS_POINT
    end

    @testset "boundary rejection, average, continuation" begin
        interior = triangle_map(0.0, 1.0, SIG, A0..., B0..., C0..., 0.5, 0.5, 0.0)
        @test interior.status == STATUS_OK
        @test abs(interior.coefficients[3, 1]) < 1e-14
        da_int = triangle_map(1.0, 1.0, SIG, A0..., B0..., C0..., 0.5, 0.5, 0.0)
        @test da_int.status == STATUS_SHEET
        @test isempty(da_int.exponents)
        @test size(da_int.coefficients, 2) == 0
        @test triangle_map(0.0, 1.0, SIG, A0..., B0..., C0..., A0...).status == STATUS_EDGE
        @test triangle_map(0.0, 1.0, SIG, A0..., B0..., C0..., 1.0, 0.0, 0.0).status == STATUS_EDGE
        @test triangle_map(2.0, 1.0, SIG, A0..., B0..., C0..., 1.0, 1.0, 0.0).status == STATUS_EDGE
        ext = triangle_map(2.0, 0.05, SIG, A0..., B0..., C0..., 3.0, 3.0, 0.0)
        @test ext.status == STATUS_OK
        @test size(ext.coefficients) == (3, 10)
        @test all(isfinite, ext.coefficients)
        @test abs(ext.coefficients[3, 1]) < 1e-12
        # Oblique exact plane, interior vs exterior
        Ao = (0.0, 0.0, 0.0)
        Bo = (4.0, 0.0, 0.0)
        Co = (0.0, 4.0, 4.0)
        @test triangle_map(1.0, 1.0, SIG, Ao..., Bo..., Co..., 1.0, 1.0, 1.0).status == STATUS_SHEET
        obl = triangle_map(1.0, 0.1, SIG, Ao..., Bo..., Co..., 4.0, 4.0, 4.0)
        @test obl.status == STATUS_OK
        @test size(obl.coefficients) == (3, 4)
    end

    @testset "radius scaling, factorials, all slots" begin
        P = (0.4, 0.3, 0.5)
        r1 = triangle_map(6.0, 1.0, SIG, A0..., B0..., C0..., P...)
        r2 = triangle_map(6.0, 0.25, SIG, A0..., B0..., C0..., P...)
        @test r1.status == STATUS_OK
        @test r2.status == STATUS_OK
        @test size(r1.coefficients) == (3, 84)
        @test length(r1.exponents) == 84
        for (k, α) in enumerate(r1.exponents)
            fac = 0.25^sum(α)
            @test relmax(r2.coefficients[:, k], r1.coefficients[:, k] .* fac) <= PREC
        end
        u = (0.3, -0.2, 0.15)
        recon = evaluate_map(r2.coefficients, r2.exponents, u)
        disp = (0.25 * u[1], 0.25 * u[2], 0.25 * u[3])
        Q = (P[1] + disp[1], P[2] + disp[2], P[3] + disp[3])
        sc = triangle_map(0.0, 1.0, SIG, A0..., B0..., C0..., Q...)
        got = (sc.coefficients[1, 1], sc.coefficients[2, 1], sc.coefficients[3, 1])
        @test nvec(sub3(recon, got)) / max(1.0, nvec(got)) < 5e-4
        # q=1 linear terms are radius * first derivatives (1! = 1)
        s0 = triangle_map(0.0, 1.0, SIG, A0..., B0..., C0..., P...)
        @test r2.coefficients[1, 1] == s0.coefficients[1, 1] ||
              abs(r2.coefficients[1, 1] - s0.coefficients[1, 1]) < 1e-14
        div, curl = maxwell_residual(r1.coefficients, 6)
        @test div < 1e-9
        @test curl < 1e-9
    end

    @testset "independent Green-kernel Taylor quadrature" begin
        P = (0.23, 0.17, 0.6)
        radius = 0.05
        r = triangle_map(4.0, radius, SIG, A0..., B0..., C0..., P...)
        @test r.status == STATUS_OK
        q96 = quadrature_jet_K(A0, B0, C0, P, 4, 24, radius)
        q48 = quadrature_jet_K(A0, B0, C0, P, 4, 16, radius)
        q96 .*= SIG / (4 * π)
        q48 .*= SIG / (4 * π)
        @test relmax(q48, q96) <= 1e-9
        @test relmax(r.coefficients, q96) <= PREC
        @test precision(BigFloat) == WORKPREC_OUTSIDE
    end

    @testset "reversal, translation, scale, oblique" begin
        P = (0.4, 0.3, 0.5)
        base = triangle_map(2.0, 0.1, SIG, A0..., B0..., C0..., P...)
        rev = triangle_map(2.0, 0.1, SIG, A0..., C0..., B0..., P...)
        @test base.status == STATUS_OK
        @test relmax(rev.coefficients, base.coefficients) <= PREC
        t = (1.5, -2.0, 0.75)
        At = add3(A0, t); Bt = add3(B0, t); Ct = add3(C0, t); Pt = add3(P, t)
        tr = triangle_map(2.0, 0.1, SIG, At..., Bt..., Ct..., Pt...)
        @test relmax(tr.coefficients, base.coefficients) <= PREC
        s = 4.0
        sc = triangle_map(2.0, 0.1 * s, SIG,
                          scale3(A0, s)..., scale3(B0, s)..., scale3(C0, s)...,
                          scale3(P, s)...)
        @test relmax(sc.coefficients, base.coefficients) <= PREC
        # proper rotation (permutation of axes with sign)
        # (x,y,z) -> (z, x, y)
        Ar = (A0[3], A0[1], A0[2])
        Br = (B0[3], B0[1], B0[2])
        Cr = (C0[3], C0[1], C0[2])
        Pr = (P[3], P[1], P[2])
        rot = triangle_map(2.0, 0.1, SIG, Ar..., Br..., Cr..., Pr...)
        # B'_x = B_z, B'_y = B_x, B'_z = B_y and monomials permute
        # Check reconstruction at a rotated increment instead of raw slots.
        u = (0.2, -0.1, 0.15)
        ur = (u[3], u[1], u[2])
        vb = evaluate_map(base.coefficients, base.exponents, u)
        vr = evaluate_map(rot.coefficients, rot.exponents, ur)
        vbr = (vb[3], vb[1], vb[2])
        @test nvec(sub3(vr, vbr)) / max(1.0, nvec(vbr)) <= PREC
        Ao = (0.0, 0.0, 0.0); Bo = (4.0, 0.0, 0.0); Co = (0.0, 4.0, 4.0)
        Po = (1.0, 1.0, 3.0)
        obl = triangle_map(0.0, 1.0, SIG, Ao..., Bo..., Co..., Po...)
        q = mu0H_from_K(quadrature_K(Ao, Bo, Co, Po, 40), SIG)
        got = (obl.coefficients[1, 1], obl.coefficients[2, 1], obl.coefficients[3, 1])
        @test nvec(sub3(got, q)) <= 1e-12 + 1e-10 * nvec(q)
    end

    @testset "malformed and nonfinite inputs" begin
        P = (0.4, 0.3, 0.5)
        @test eval_standalone_tokens(String[]).status == STATUS_INPUT
        @test eval_standalone_tokens(["x"]).status == STATUS_INPUT
        @test triangle_map(7.0, 1.0, SIG, A0..., B0..., C0..., P...).status == STATUS_ORDER
        @test triangle_map(1.5, 1.0, SIG, A0..., B0..., C0..., P...).status == STATUS_ORDER
        @test triangle_map(-1.0, 1.0, SIG, A0..., B0..., C0..., P...).status == STATUS_ORDER
        @test triangle_map(0.0, 0.0, SIG, A0..., B0..., C0..., P...).status == STATUS_RADIUS
        @test triangle_map(0.0, -0.2, SIG, A0..., B0..., C0..., P...).status == STATUS_RADIUS
        @test triangle_map(0.0, Inf, SIG, A0..., B0..., C0..., P...).status == STATUS_RADIUS
        @test triangle_map(0.0, 1.0, SIG, A0..., B0..., C0..., Inf, 0.0, 0.0).status == STATUS_POINT
        @test triangle_map(0.0, 1.0, SIG, A0..., B0..., C0..., NaN, 0.0, 0.0).status == STATUS_POINT
        @test triangle_map(0.0, 1.0, SIG, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0, P...).status == STATUS_INPUT
        @test triangle_map(0.0, 1.0, NaN, A0..., B0..., C0..., P...).status == STATUS_INPUT
        @test triangle_map(0.0, 1.0, SIG, NaN, 0.0, 0.0, B0..., C0..., P...).status == STATUS_INPUT
        r = triangle_map(0.0, 1.0, SIG, A0..., B0..., C0..., P...)
        @test r.status == STATUS_OK
        @test all(isfinite, r.coefficients)
        @test precision(BigFloat) == WORKPREC_OUTSIDE
        setprecision(BigFloat, 128) do
            triangle_map(1.0, 0.2, SIG, A0..., B0..., C0..., P...)
            @test precision(BigFloat) == 128
        end
        @test precision(BigFloat) == WORKPREC_OUTSIDE
    end

    @testset "CLI JSON and batch" begin
        args = ["0", "1", "0.8", "0", "0", "0", "2", "0", "0", "0", "2", "0", "0.4", "0.3", "0.5"]
        out, ok = run_cli(args)
        @test ok
        @test parse_json_status(out) == 0
        @test occursin("\"exponents\":[[0,0,0]]", replace(out, ' ' => "")) ||
              occursin("\"exponents\":[[0, 0, 0]]", out) ||
              occursin("[0,0,0]", out)
        @test !occursin("NaN", out)
        @test !occursin("Inf", out)
        bad = ["7", "1", "0.8", "0", "0", "0", "2", "0", "0", "0", "2", "0", "0.4", "0.3", "0.5"]
        iob = IOBuffer()
        proc = run(pipeline(ignorestatus(`$(JULIA) --startup-file=no $(joinpath(PKG, "cli.jl")) $bad`), iob))
        outb = String(take!(iob))
        @test parse_json_status(outb) == STATUS_ORDER
        @test occursin("\"coefficients\":[]", replace(outb, ' ' => ""))
        @test proc.exitcode != 0
        body = join((join(args, ' '), join(bad, ' ')), '\n') * '\n'
        bout = run_cli_batch(body)
        lines = filter(!isempty, split(bout, '\n'))
        @test length(lines) == 2
        @test parse_json_status(lines[1]) == 0
        @test parse_json_status(lines[2]) == STATUS_ORDER
    end

    @testset "native Fortran client" begin
        worktree = normpath(joinpath(PKG, ".."))
        srcdir = joinpath(worktree, "fortran", "native")
        exactc = joinpath(worktree, "fortran", "triangle_exact_geometry.c")
        build = mktempdir()
        lib = joinpath(build, "libtan.so")
        cache = joinpath(build, "triangle.tan")
        built = false
        if isfile(joinpath(srcdir, "tan_mod.f90")) && isfile(exactc)
            try
                run(Cmd(`gcc -O3 -fno-fast-math -std=c11 -fPIC -c $exactc -o exact.o`; dir=build))
                run(Cmd(`gfortran -O3 -fno-fast-math -std=f2018 -fPIC -ffree-line-length-none -c $(joinpath(srcdir, "tan_jet.f90"))`; dir=build))
                run(Cmd(`gfortran -O3 -fno-fast-math -std=f2018 -fPIC -ffree-line-length-none -c $(joinpath(srcdir, "tan_mod.f90"))`; dir=build))
                run(Cmd(`gfortran -O3 -fno-fast-math -fPIC -shared -o libtan.so tan_jet.o tan_mod.o exact.o -lgmp`; dir=build))
                built = isfile(lib)
            catch err
                @info "native library build skipped" exception=err
            end
        end
        @test built
        open(cache, "w") do io
            println(io, "TAN_CACHE_V1")
            println(io, "1 3")
            println(io, "0 0 0 0 0 1")
            println(io, "1 0 0 0 2 0 0 0.8")
            println(io, "1 2 0 0 0 2 0 0.8")
            println(io, "1 0 2 0 0 0 0 0.8")
        end
        nr = native_eval(lib, cache, 0.4, 0.3, 0.5, 1, 0, 1.0)
        @test nr.status == 0
        @test size(nr.coefficients) == (3, 1)
        sr = triangle_map(0.0, 1.0, 0.8, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.4, 0.3, 0.5)
        @test relmax(nr.coefficients, sr.coefficients) <= PREC
        nq = native_eval(lib, cache, 0.4, 0.3, 0.5, 1, 2, 0.1)
        @test nq.status == 0
        @test size(nq.coefficients) == (3, 10)
        @test native_eval(lib, cache, 0.0, 0.0, 0.0, 1, 0, 1.0).status == STATUS_EDGE
        @test native_eval(lib, cache, 0.5, 0.5, 0.0, 1, 1, 0.1).status == STATUS_SHEET
        @test native_eval(lib, cache, 0.4, 0.3, 0.5, 1, 7, 1.0).status == STATUS_ORDER
        iob = IOBuffer()
        ncli = joinpath(PKG, "native_cli.jl")
        proc = run(pipeline(ignorestatus(`$(JULIA) --startup-file=no $ncli $lib $cache 0.4 0.3 0.5 1 0 1.0`), iob))
        nout = String(take!(iob))
        @test proc.exitcode == 0
        @test parse_json_status(nout) == 0
        @test occursin("Shares the Fortran tan_mod physical kernel",
                       read(`$(JULIA) --startup-file=no $ncli --help`, String))
    end
end

@testset "native input range and failed reload" begin
    @test TriangularPMFields.eval_native_tokens(["missing", "cache", "0", "0", "1", "1e100", "0", "1"]).status == 4
    @test TriangularPMFields.eval_native_tokens(["missing", "cache", "0", "0", "1", "1", "1e100", "1"]).status == STATUS_ORDER
    @test native_eval(TriangularPMFields.NATIVE.libpath, TriangularPMFields.NATIVE.cache, 0.4, 0.3, 0.5, 1, 0, 1.0).status == 0
    @test native_last_mode() == 1
    @test native_load!(TriangularPMFields.NATIVE.libpath, "missing-cache-path") == 2
    @test native_last_mode() == -1
end

@testset "standalone malformed input and map dimensions" begin
    valid = ["0", "1", "1", "0", "0", "0", "1", "0", "0", "0", "1", "0", ".25", ".25", ".75"]
    huge = copy(valid); huge[1] = "1e100"
    negative = copy(valid); negative[1] = "-1e100"
    @test eval_standalone_tokens(huge).status == STATUS_ORDER
    @test eval_standalone_tokens(negative).status == STATUS_ORDER
    output = run_cli_batch(join([join(huge,' '), join(negative,' '), join(valid,' ')], '\n') * '\n')
    rows = filter(!isempty, split(output,'\n'))
    @test length(rows) == 3
    @test [parse_json_status(row) for row in rows] == [STATUS_ORDER, STATUS_ORDER, 0]
    @test_throws DimensionMismatch evaluate_map(zeros(3,1), exponents(6), (0.,0.,0.))
    @test_throws DimensionMismatch evaluate_map(zeros(2,1), exponents(0), (0.,0.,0.))
end
