using Pkg
Pkg.activate(".")
Pkg.instantiate()

using JuMP, Gurobi, JSON, Distributions, LazySets, Polyhedra, CDDLib, Dualization, LinearAlgebra, Statistics, JLD, Plots, LaTeXStrings
using GLM, DataFrames, Random, NearestNeighbors, StatsPlots
const GUROBI_ENV = Gurobi.Env()


data_file = joinpath(pwd(), "parameters.json")

data = JSON.parsefile(data_file)

function save_json(output_dict::Dict, path::String, output_name::String="output")
    jD = JSON.json(output_dict)

    open(joinpath(path, "$output_name.json"), "w") do j
        write(j, jD)
    end

    return
end

function open_json(input_file::String, path::String)
    file_data = JSON.parsefile(joinpath(path, input_file))
    return file_data
end

function scenarios(data, D)

    Random.seed!(2025)

    sample = collect(1:data["sample_size"])
    p = 3  

    ξ_dict = Dict("zstar" => zeros(p), "PredictionStar" => zeros(D), "z" => [zeros(p) for n ∈ sample], "y" => [zeros(D) for n ∈ sample], "Ξ_y(z^*)" => zeros(D, 2))

    # contexts
    z = [trunc.(rand(Normal(1,0.5), p), digits=2) for n ∈ sample]
    ξ_dict["z"] = z

    β0 = fill(10.0, D) 
    β = [rand(p).* 2.0 for d ∈ collect(1:D)]

    σ = 2.0

    ξ = [zeros(D) for n ∈ sample]

    for d ∈ collect(1:D)
        for n ∈ sample

            ξ[n][d]  = β0[d] + dot(β[d],z[n]) + rand(Normal(0,σ))

        end
    end

    # Predictor

    Models = Vector{Any}(undef,D)

    ξ_forcast = [zeros(D) for n ∈ sample]

    for d ∈ collect(1:D)

        df = DataFrame(
            z1 = [z[n][1] for n ∈ sample],
            z2 = [z[n][2] for n ∈ sample],
            z3 = [z[n][3] for n ∈ sample],
            y = [ξ[n][d] for n ∈ sample]
        )

        model = lm(@formula(y ~ z1 + z2 + z3),df)

        Models[d] = model
        Forcast = predict(model)
        for n ∈ sample
            ξ_forcast[n][d] = Forcast[n]
        end

    end

    ε = ξ - ξ_forcast

    Z = [z[n][i] for n ∈ sample, i ∈ collect(1:p)]
    Residuals = [trunc(ε[n][d], digits=2) for n ∈ sample, d ∈ collect(1:D)]

    tree = KDTree(permutedims(Z))

    zstar = trunc.(rand(Normal(1,0.5), p), digits=2)

    k = 50

    idxs, dists = knn(tree,zstar,k,true)

    LocalResiduals = Residuals[idxs,:]

    # Contextual dependent uncertainty set Ξ_y(z^*)

    Lower = zeros(D)

    Upper = zeros(D)

    α = 0.05

    for d in 1:D

        Lower[d] = quantile(LocalResiduals[:,d],α)

        Upper[d] = quantile(LocalResiduals[:,d],1-α)

    end

    # Prediction for z^*

    PredictionStar = zeros(D)

    for d in 1:D

        PredictionStar[d] = predict(
            Models[d],
            DataFrame(
                z1=[zstar[1]],
                z2=[zstar[2]],
                z3=[zstar[3]]
            )
        )[1]

    end

    ξ_dict["zstar"] = zstar
    ξ_dict["PredictionStar"] = trunc.(PredictionStar, digits=2)
    for n ∈ sample
        for d ∈ collect(1:D)
            ξ_dict["y"][n][d] = trunc(ε[n][d], digits=2)
        end
    end
    ξ_dict["Ξ_y(z^*)"] = Matrix([Lower[:] Upper[:]])

    return ξ_dict
   
end

function recourse(data, instalations, demand, x, G, D)

    deficit = data["fixed_cost"]["deficit"]
    storage = data["fixed_cost"]["storage"]
    cost = [norm([instalations["$G"][g], demand["$D"][d]], 2) for g ∈ 1:G, d ∈ 1:D]

    model =  Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0))
    @variables(model, begin
        y[1:G, 1:D] ≥ 0
        u[1:D] ≥ 0
        v[1:G] ≥ 0
    end)

    @constraints(model, begin
        σ[g ∈ 1:G], sum(y[g, d] for d ∈ 1:D) + v[g] == x[g]
        π[d ∈ 1:D], sum(y[g, d] for g ∈ 1:G) + u[d] ≥ 0
    end)

    @objective(model, Min, sum(cost[g, d]*y[g, d] for g ∈ 1:G, d ∈ 1:D) + deficit*sum(u[d] for d ∈ 1:D) + storage*sum(v[g] for g ∈ 1:G))

    return model
    
end

function recourse_test(data, instalations, demand, x, G, D, ξ_dict)

    deficit = data["fixed_cost"]["deficit"]
    storage = data["fixed_cost"]["storage"]
    cost = [norm([instalations["$G"][g], demand["$D"][d]], 2) for g ∈ 1:G, d ∈ 1:D]

    y_scen = ξ_dict["y"]
    y_mean = mean(y_scen)
    PredictionStar = ξ_dict["PredictionStar"]

    model =  Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0))
    @variables(model, begin
        y[1:G, 1:D] ≥ 0
        u[1:D] ≥ 0
        v[1:G] ≥ 0
    end)

    @constraints(model, begin
        σ[g ∈ 1:G], sum(y[g, d] for d ∈ 1:D) + v[g] == x[g]
        π[d ∈ 1:D], sum(y[g, d] for g ∈ 1:G) + u[d] ≥ PredictionStar[d] + y_mean[d]
    end)

    @objective(model, Min, sum(cost[g, d]*y[g, d] for g ∈ 1:G, d ∈ 1:D) + deficit*sum(u[d] for d ∈ 1:D) + storage*sum(v[g] for g ∈ 1:G))

    return model
    
end

function create_oracle_problem(data, instalations, demand, G, D)

    x = zeros(G)
    v = zeros(G)
    model = recourse(data, instalations, demand, x, G, D)
    m_oracle = dualize(model; dual_names = DualNames())
   
    @variable(m_oracle, ηpos[d ∈ 1:D], Bin)
    @variable(m_oracle, ηneg[d ∈ 1:D], Bin)
    @constraint(m_oracle, [d ∈ 1:D], ηpos[d] + ηneg[d] ≤ 1)
    set_optimizer(m_oracle, optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0))
    obj = objective_function(m_oracle)

    return m_oracle, obj

end

function modify_oracle(m_oracle, obj, ξ_dict, n, x, λ, G, D)

    y = ξ_dict["y"][n]
    Ξ_y = ξ_dict["Ξ_y(z^*)"]
    PredictionStar = ξ_dict["PredictionStar"]  

    ex1 =  @expression(m_oracle, QuadExpr())
    ex2 =  @expression(m_oracle, AffExpr())
    ex3 =  @expression(m_oracle, AffExpr())

    for d ∈ 1:D
        Δpos = Ξ_y[d, 2] - y[d]
        Δneg = y[d] - Ξ_y[d, 1]    
        add_to_expression!(ex3, y[d], m_oracle[Symbol("π[$d]")])
        add_to_expression!(ex3, PredictionStar[d], m_oracle[Symbol("π[$d]")])
        add_to_expression!(ex1, Δpos, m_oracle[Symbol("π[$d]")], m_oracle[:ηpos][d])
        add_to_expression!(ex1, -Δneg, m_oracle[Symbol("π[$d]")], m_oracle[:ηneg][d])
        add_to_expression!(ex3, -λ*Δpos,  m_oracle[:ηpos][d])
        add_to_expression!(ex3, -λ*Δneg,  m_oracle[:ηneg][d])
    end

    for g ∈ 1:G
        add_to_expression!(ex2, x[g], m_oracle[Symbol("σ[$g]")])
    end

     @objective(m_oracle, Max, obj + ex1 + ex2 + ex3)
    
end

function solve_oracle_problem(data, m_oracle, obj, ξ_dict, x, λ, μ, θ, G, D, tol, y_dict, π_dict, σ_dict, L, norm_l1_y)

    y = ξ_dict["y"]
    Ξ_y = ξ_dict["Ξ_y(z^*)"]
    zstar = ξ_dict["zstar"]
    z = ξ_dict["z"]


    for n ∈ 1:data["sample_size"]
       
        if tol[n]        
            val, time = @timed(begin
                modify_oracle(m_oracle, obj, ξ_dict, n, x, λ, G, D)
                optimize!(m_oracle)
                status = termination_status(m_oracle)
                if status != MOI.OPTIMAL
                    error(" Status: $(status)")
                end
            end)
        end
        obj_val = objective_value(m_oracle)
        if μ[n] + θ - obj_val - λ*norm(zstar - z[n], 1) ≥ -1e-2
            tol[n] = false
        else
            Δpos = Ξ_y[:, 2] - y[n]
            Δneg = y[n] - Ξ_y[:, 1]    
            ηpos = value.(m_oracle[:ηpos])
            ηneg = value.(m_oracle[:ηneg])
            π_opt = [value(m_oracle[Symbol("π[$d]")]) for d ∈ 1:D]
            σ_opt = [value(m_oracle[Symbol("σ[$g]")]) for g ∈ 1:G]
            y_opt = y[n] + ηpos.*Δpos - ηneg.*Δneg
            push!(L["$n"], length(L["$n"]) + 1)
            push!(y_dict["$n"], y_opt)
            push!(π_dict["$n"], π_opt)
            push!(σ_dict["$n"], σ_opt)
            push!(norm_l1_y["$n"], norm(y_opt - y[n], 1))
        end
    end
    
end

function create_master_problem(data, G)

    N = data["sample_size"]
    capacity = data["capacity"]

    m_master = Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0))
    @variables(m_master, begin
        x[1:G] ≥ 0
        μ[1:N] ≥ 0
        θ
        λ ≥ 0
    end)
    @constraints(m_master, begin
        [g ∈ 1:G], x[g] ≤ capacity
    end)

    @objective(m_master, Min, m_master[:λ]*ρ + m_master[:θ] + (1/(N*α))*sum(m_master[:μ]))

     return m_master
end

function update_master_CCG(data, m_master, ξ_dict, y_dict, norm_l1_y, G, D, tol)

    N = data["sample_size"]

    deficit = data["fixed_cost"]["deficit"]
    storage = data["fixed_cost"]["storage"]
    cost = [norm([instalations["$G"][g], demand["$D"][d]], 2) for g ∈ 1:G, d ∈ 1:D]

    zstar = ξ_dict["zstar"]
    PredictionStar = ξ_dict["PredictionStar"]
    z = ξ_dict["z"]

    for n ∈ 1:N

        norm_l1_z = norm(zstar - z[n], 1)

        if tol[n]

            y_opt = y_dict["$n"][end]

            y = @variable(m_master, [1:G, 1:D], lower_bound = 0)
            u = @variable(m_master, [1:D], lower_bound = 0)
            v = @variable(m_master, [1:G], lower_bound = 0)
            @constraints(m_master, begin
                        [g ∈ 1:G], sum(y[g, d] for d ∈ 1:D) + v[g] == m_master[:x][g]
                        [d ∈ 1:D], sum(y[g, d] for g ∈ 1:G) + u[d] ≥ PredictionStar[d] + y_opt[d]
            end)
            @constraint(m_master, sum(cost[g, d]*y[g, d] for g ∈ 1:G, d ∈ 1:D) + deficit*sum(u[d] for d ∈ 1:D) + storage*sum(v[g] for g ∈ 1:G) - m_master[:λ]*norm_l1_y["$n"][end] - m_master[:λ]*norm_l1_z ≤ m_master[:μ][n] + m_master[:θ])
        end

    end

    
end

function update_master_Benders(data, m_master, ξ_dict, y_dict, π_dict, σ_dict, norm_l1_y, G, D, tol)

    N = data["sample_size"]

    zstar = ξ_dict["zstar"]
    PredictionStar = ξ_dict["PredictionStar"]
    z = ξ_dict["z"]

    for n ∈ 1:N

        norm_l1_z = norm(zstar - z[n], 1)

        if tol[n]

            y_opt = y_dict["$n"][end]
            π_opt = π_dict["$n"][end]
            σ_opt = σ_dict["$n"][end]

            @constraint(m_master, sum(σ_opt[g]*m_master[:x][g] for g ∈ 1:G) + sum(π_opt[d]*y_opt[d] for d ∈ 1:D) + sum(PredictionStar[d]*π_opt[d] for d ∈ 1:D) - m_master[:λ]*norm_l1_y["$n"][end] - m_master[:λ]*norm_l1_z ≤ m_master[:μ][n] + m_master[:θ])
        end

    end

    
end

function initilization_restricted_master_CCG(data, m_master, ξ_dict)

    N = data["sample_size"]

    deficit = data["fixed_cost"]["deficit"]
    storage = data["fixed_cost"]["storage"]
    cost = [norm([instalations["$G"][g], demand["$D"][d]], 2) for g ∈ 1:G, d ∈ 1:D]

    zstar = ξ_dict["zstar"]
    PredictionStar = ξ_dict["PredictionStar"]
    z = ξ_dict["z"]
    y_scen = ξ_dict["y"]

    for n ∈ 1:N

        y_mean = mean(y_scen)
        norm_l1_z = norm(zstar - z[n], 1)
        norm_l1_y = norm(y_mean - y_scen[n], 1)

        y = @variable(m_master, [1:G, 1:D], lower_bound = 0)
        u = @variable(m_master, [1:D], lower_bound = 0)
        v = @variable(m_master, [1:G], lower_bound = 0)
        @constraints(m_master, begin
                    [g ∈ 1:G], sum(y[g, d] for d ∈ 1:D) + v[g] == m_master[:x][g]
                    [d ∈ 1:D], sum(y[g, d] for g ∈ 1:G) + u[d] ≥ PredictionStar[d] + y_mean[d]
        end)
        @constraint(m_master, sum(cost[g, d]*y[g, d] for g ∈ 1:G, d ∈ 1:D) + deficit*sum(u[d] for d ∈ 1:D) + storage*sum(v[g] for g ∈ 1:G) - m_master[:λ]*norm_l1_y - m_master[:λ]*norm_l1_z ≤ m_master[:μ][n] + m_master[:θ])

    end

    return nothing
 
end

function solve_master_problem(m_master)

    set_optimizer_attribute(m_master, "DualReductions", 0)

    optimize!(m_master)
    status = termination_status(m_master)
    if status != MOI.OPTIMAL
        error("problem $status")
    end

    # Solution
    x = value.(m_master[:x])
    λ = value.(m_master[:λ])

    μ = value.(m_master[:μ])
    θ = value(m_master[:θ])

    return x, λ, μ, θ
    
end

function critical_radius(data, ξ_dict, α; p = 1)

    N = data["sample_size"]

    zstar = ξ_dict["zstar"]
    z = ξ_dict["z"]
    y = ξ_dict["y"]

    Ξ_y = ξ_dict["Ξ_y(z^*)"]
    a = Ξ_y[:,1]
    b = Ξ_y[:,2]

    dist = zeros(N)

    for n in 1:N

        dz = norm(z[n] - zstar, 1)

        dy = 0.0
        for d in eachindex(a)
            if y[n][d] < a[d]
                dy += a[d] - y[n][d]
            elseif y[n][d] > b[d]
                dy += y[n][d] - b[d]
            end
        end

        dist[n] = dz + dy
    end

    sort!(dist)

    if α == 0
        return dist[1]
    end

    m = floor(Int, N*α)
    frac = N*α - m

    value = sum(dist[k] for k in 1:m)

    if frac > 0
        value += frac*dist[m+1]
    end

    return value/(N*α)

end

# Instalations
#  instalations = Dict("$G" => [(rand(Uniform(0,1)), rand(Uniform(0,1))) for _ ∈ 1:G] for G in data["instalations"])
#  save_json(instalations, pwd(), "instalations_dict")
instalations = open_json("instalations_dict.json", pwd())

# Uncertainty
#  demand = Dict("$D" => [(rand(Uniform(0,1)), rand(Uniform(0,1))) for _ ∈ 1:D] for D in data["demand"])
#   save_json(demand, pwd(), "demand_dict")
demand = open_json("demand_dict.json", pwd())

# ξ_dict = open_json("ξ_dict.json", pwd())

G, D = 20, 50

ξ_dict = scenarios(data, D)

α = 0.5
sample = collect(1:data["sample_size"])

ρ = critical_radius(data, ξ_dict, α) + 10

PredictionStar = ξ_dict["PredictionStar"]
y_scen = ξ_dict["y"]

Dmax = maximum([sum(PredictionStar[d] + y_scen[n][d] for d ∈ 1:D) for n ∈ sample])
capacity = Dmax/(G - 2)

# CCG Algorithm

y_dict = Dict("$n" => [] for n ∈ sample)
π_dict = Dict("$n" => [] for n ∈ sample) 
σ_dict = Dict("$n" => [] for n ∈ sample)
L = Dict("$n" => [] for n ∈ sample) 
norm_l1_y = Dict("$n" => [] for n ∈ sample) 
tol = [true for n ∈ sample]
iter = 1
m_master = create_master_problem(data, G)
initilization_restricted_master_CCG(data, m_master, ξ_dict)
m_oracle, obj  = create_oracle_problem(data, instalations, demand, G, D)
while Base.any(tol) && iter ≤ 100
    global x_CCG, λ_CCG, μ_CCG, θ_CCG = solve_master_problem(m_master)
    solve_oracle_problem(data, m_oracle, obj, ξ_dict, x_CCG, λ_CCG, μ_CCG, θ_CCG, G, D, tol, y_dict, π_dict, σ_dict, L, norm_l1_y)
    update_master_CCG(data, m_master, ξ_dict, y_dict, norm_l1_y, G, D, tol)
    iter +=  1
end

# Benders Multi-Cut Algorithm

y_dict = Dict("$n" => [] for n ∈ sample) 
π_dict = Dict("$n" => [] for n ∈ sample) 
σ_dict = Dict("$n" => [] for n ∈ sample)
L = Dict("$n" => [] for n ∈ sample) 
norm_l1_y = Dict("$n" => [] for n ∈ sample) 
tol = [true for n ∈ sample]
iter = 1
m_master = create_master_problem(data, G)
@constraint(m_master, m_master[:θ] ≥ -1e4)
m_oracle, obj  = create_oracle_problem(data, instalations, demand, G, D)
while Base.any(tol) && iter ≤ 100
    global x_Benders, λ_Benders, μ_Benders, θ_Benders = solve_master_problem(m_master)
    solve_oracle_problem(data, m_oracle, obj, ξ_dict, x_Benders, λ_Benders, μ_Benders, θ_Benders, G, D, tol, y_dict, π_dict, σ_dict, L, norm_l1_y)
    update_master_Benders(data, m_master, ξ_dict, y_dict, π_dict, σ_dict, norm_l1_y, G, D, tol)
    iter +=  1
end
