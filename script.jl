using Pkg
Pkg.activate(".")
Pkg.instantiate()

using JuMP, Gurobi, JSON, Distributions, LazySets, Polyhedra, CDDLib, Dualization, LinearAlgebra, Statistics, JLD, Plots, LaTeXStrings
using GLM, DataFrames, Random, NearestNeighbors, StatsPlots
const GUROBI_ENV = Gurobi.Env()

data_file = joinpath(pwd(), "parameters.json")

data = JSON.parsefile(data_file)

Random.seed!(2025)

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

function scenarios(data, D, ξ_dict, p)

    sample = collect(1:data["sample_size"])

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
            # μ = β0[d] + dot(β[d], z[n])
            # ξ[n][d] = rand(LogNormal(μ, σ))

        end
    end

    # Predictor

    Models = Vector{Any}(undef,D)

    ξ_forcast = [zeros(D) for n ∈ sample]

    for d ∈ collect(1:D)

        df = DataFrame()
        for i in 1:p
            df[!, Symbol("z$i")] = [z[n][i] for n in sample]
        end
        df[!, :y] = [ξ[n][d] for n in sample]

        model = lm(term(:y) ~ sum(term(Symbol("z$i")) for i in 1:p), df)

        Models[d] = model
        Forcast = predict(model)
        for n ∈ sample
            ξ_forcast[n][d] = Forcast[n]
        end

    end

    ε = ξ - ξ_forcast

    for n ∈ sample
        for d ∈ collect(1:D)
            ξ_dict["y"][n][d] = trunc(ε[n][d], digits=2)
        end
    end

    return Models
   
end

function forecast(D, zstar, Models)

    # Prediction for z^*

    PredictionStar = zeros(D)

    for d in 1:D

        PredictionStar[d] = predict(
            Models[d],
            DataFrame([Symbol("z$i") => [zstar[i]] for i in 1:p])
        )[1]

    end

    PredictionStar = trunc.(PredictionStar, digits=2)

    return PredictionStar

end

function uncertainty_set(ξ_dict, zstar)

    z = ξ_dict["z"]
    y = ξ_dict["y"]

    p = length(z[1])
    D = length(y[1])

    Z = [z[n][i] for n ∈ sample, i ∈ collect(1:p)]
    Residuals = [y[n][d] for n ∈ sample, d ∈ collect(1:D)]

    tree = KDTree(permutedims(Z))

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

    Ξ_y = Matrix([Lower[:] Upper[:]])

    return Ξ_y

    
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

function create_oracle_problem(data, instalations, demand, G, D)

    x = zeros(G)
    model = recourse(data, instalations, demand, x, G, D)
    m_oracle = dualize(model; dual_names = DualNames())
   
    @variable(m_oracle, ηpos[d ∈ 1:D], Bin)
    @variable(m_oracle, ηneg[d ∈ 1:D], Bin)
    @constraint(m_oracle, [d ∈ 1:D], ηpos[d] + ηneg[d] ≤ 1)
    set_optimizer(m_oracle, optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0))
    obj = objective_function(m_oracle)

    return m_oracle, obj

end

function modify_oracle(m_oracle, obj, ξ_dict, Ξ_y, n, x, λ, G, D, PredictionStar)

    y = ξ_dict["y"][n]

    ex1 = @expression(m_oracle, QuadExpr())
    ex2 = @expression(m_oracle, AffExpr())
    ex3 = @expression(m_oracle, AffExpr())

    for d in 1:D

        a = Ξ_y[d,1]
        b = Ξ_y[d,2]

        yproj = clamp(y[d], a, b)

        Δpos = b - yproj
        Δneg = yproj - a

        π = m_oracle[Symbol("π[$d]")]

        add_to_expression!(ex3, PredictionStar[d], π)
        add_to_expression!(ex3, yproj, π)

        add_to_expression!(ex3, -λ*abs(y[d]-yproj))

        add_to_expression!(ex1, Δpos, π, m_oracle[:ηpos][d])
        add_to_expression!(ex1, -Δneg, π, m_oracle[:ηneg][d])

        add_to_expression!(ex3, -λ*Δpos, m_oracle[:ηpos][d])
        add_to_expression!(ex3, -λ*Δneg, m_oracle[:ηneg][d])

    end

    for g in 1:G
        add_to_expression!(ex2, x[g], m_oracle[Symbol("σ[$g]")])
    end

    @objective(m_oracle, Max, obj + ex1 + ex2 + ex3)

end

function solve_oracle_problem(data, m_oracle, obj, ξ_dict, Ξ_y, x, λ, μ, θ, G, D, tol, y_dict, y_old, π_dict, σ_dict, L, violation, zstar, PredictionStar)

    y = ξ_dict["y"]
    z = ξ_dict["z"]
    a = Ξ_y[:,1]
    b = Ξ_y[:,2]

    for n ∈ 1:data["sample_size"]
       
        if tol[n]        
            val, time = @timed(begin
                modify_oracle(m_oracle, obj, ξ_dict, Ξ_y, n, x, λ, G, D, PredictionStar)
                optimize!(m_oracle)
                status = termination_status(m_oracle)
                if status != MOI.OPTIMAL
                    error(" Status: $(status)")
                end
            end)
        end
        obj_val = objective_value(m_oracle)
        viol = μ[n] + θ - obj_val + λ*norm(zstar - z[n],1)
        if viol ≥ -1e-2
            tol[n] = false
        else
            push!(violation["$n"], viol)
            yproj = clamp.(y[n], a, b)  
            Δpos = b .- yproj
            Δneg = yproj .- a
            ηpos = value.(m_oracle[:ηpos])
            ηneg = value.(m_oracle[:ηneg])
            π_opt = [value(m_oracle[Symbol("π[$d]")]) for d ∈ 1:D]
            σ_opt = [value(m_oracle[Symbol("σ[$g]")]) for g ∈ 1:G]
            y_opt = yproj .+ ηpos .* Δpos .- ηneg .* Δneg
            push!(L["$n"], length(L["$n"]) + 1)
            push!(y_dict["$n"], y_opt)
            push!(π_dict["$n"], π_opt)
            push!(σ_dict["$n"], σ_opt)
            dist = norm(y_opt - y[n],1) + norm(zstar - z[n],1)
            score = -viol/(1 + dist)
            if score > y_old["$n"]["best_score"]
                y_old["$n"]["best_score"] = score
                y_old["$n"]["best_vertice"] = copy(y_opt)
                y_old["$n"]["best_π"] = copy(π_opt)
                y_old["$n"]["best_σ"] = copy(σ_opt)
            end
        end
    end
    
end

function create_master_problem(data, G, ρ)

    N = data["sample_size"]
    capacity = data["capacity"]

    m_master = Model(
    optimizer_with_attributes(
            () -> Gurobi.Optimizer(GUROBI_ENV),
            "NumericFocus" => 3,
            "ScaleFlag" => 2,
            "Method" => 1,
            "OutputFlag" => 0
        )
    )
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

function update_master_CCG(data, m_master, ξ_dict, y_dict, G, D, tol, zstar, PredictionStar)

    N = data["sample_size"]

    deficit = data["fixed_cost"]["deficit"]
    storage = data["fixed_cost"]["storage"]
    cost = [norm([instalations["$G"][g], demand["$D"][d]], 2) for g ∈ 1:G, d ∈ 1:D]
    z = ξ_dict["z"]
    y_scen = ξ_dict["y"]

    for n ∈ 1:N

        norm_l1_z = norm(zstar - z[n], 1)

        if tol[n]

            y_opt = y_dict["$n"][end]
            norm_l1_y = norm(y_opt - y_scen[n], 1)

            y = @variable(m_master, [1:G, 1:D], lower_bound = 0)
            u = @variable(m_master, [1:D], lower_bound = 0)
            v = @variable(m_master, [1:G], lower_bound = 0)
            @constraints(m_master, begin
                        [g ∈ 1:G], sum(y[g, d] for d ∈ 1:D) + v[g] == m_master[:x][g]
                        [d ∈ 1:D], sum(y[g, d] for g ∈ 1:G) + u[d] ≥ PredictionStar[d] + y_opt[d]
            end)
            @constraint(m_master, sum(cost[g, d]*y[g, d] for g ∈ 1:G, d ∈ 1:D) + deficit*sum(u[d] for d ∈ 1:D) + storage*sum(v[g] for g ∈ 1:G) - m_master[:λ]*norm_l1_y - m_master[:λ]*norm_l1_z ≤ m_master[:μ][n] + m_master[:θ])
        end

    end

    
end

function update_master_Benders(data, m_master, ξ_dict, y_dict, π_dict, σ_dict, G, D, tol, zstar, PredictionStar)

    N = data["sample_size"]

    z = ξ_dict["z"]
    y_scen = ξ_dict["y"]

    for n ∈ 1:N

        norm_l1_z = norm(zstar - z[n], 1)

        if tol[n]

            y_opt = y_dict["$n"][end]
            π_opt = π_dict["$n"][end]
            σ_opt = σ_dict["$n"][end]

            norm_l1_y = norm(y_opt - y_scen[n], 1)

            @constraint(m_master, sum(σ_opt[g]*m_master[:x][g] for g ∈ 1:G) + sum(π_opt[d]*y_opt[d] for d ∈ 1:D) + sum(PredictionStar[d]*π_opt[d] for d ∈ 1:D) - m_master[:λ]*norm_l1_y - m_master[:λ]*norm_l1_z ≤ m_master[:μ][n] + m_master[:θ])
        end

    end

    
end

function initilization_restricted_master_CCG(data, instalations, demand, G, D, m_master, ξ_dict, Ξ_y, zstar, PredictionStar, α)

    N = data["sample_size"]

    deficit = data["fixed_cost"]["deficit"]
    storage = data["fixed_cost"]["storage"]
    cost = [norm([instalations["$G"][g], demand["$D"][d]], 2) for g ∈ 1:G, d ∈ 1:D]

    z = ξ_dict["z"]
    y_scen = ξ_dict["y"]
    a = Ξ_y[:,1]
    b = Ξ_y[:,2]

    m = ceil(Int, N*α)

    dist = zeros(N)

    for n in 1:N

        dz = norm(z[n] - zstar, 1)

        dy = 0.0
        for d in 1:D
            if y_scen[n][d] < a[d]
                dy += a[d] - y_scen[n][d]
            elseif y_scen[n][d] > b[d]
                dy += y_scen[n][d] - b[d]
            end
        end

        dist[n] = dz + dy

    end

    idx = sortperm(dist)

    for k in 1:m

        n = idx[k]

        yproj = clamp.(y_scen[n], a, b)

        norm_l1_z = norm(zstar - z[n], 1)
        norm_l1_y = norm(yproj - y_scen[n], 1)

        y = @variable(m_master, [1:G,1:D], lower_bound = 0)
        u = @variable(m_master, [1:D], lower_bound = 0)
        v = @variable(m_master, [1:G], lower_bound = 0)

        @constraints(m_master, begin
            [g=1:G], sum(y[g,d] for d=1:D) + v[g] == m_master[:x][g]
            [d=1:D], sum(y[g,d] for g=1:G) + u[d] >= PredictionStar[d] + yproj[d]
        end)

        @constraint(
            m_master,
            sum(cost[g,d]*y[g,d] for g=1:G,d=1:D)
            + deficit*sum(u)
            + storage*sum(v)
            - m_master[:λ]*(norm_l1_y + norm_l1_z)
            <= m_master[:μ][n] + m_master[:θ]
        )

    end

    return nothing
 
end

function deterministic_equivalent_problem(data, instalations, demand, G, D, y_list, PredictionStar)

    m = length(y_list)
    capacity = data["capacity"]
    deficit = data["fixed_cost"]["deficit"]
    storage = data["fixed_cost"]["storage"]
    cost = [norm([instalations["$G"][g], demand["$D"][d]], 2) for g ∈ 1:G, d ∈ 1:D]


    model =  Model(optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0))
    @variables(model, begin
        x[1:G] ≥ 0
        cost_recourse[1:m]
    end)

    @constraint(model, [g ∈ 1:G], x[g] ≤ capacity)

    for k in 1:m

        y_opt = y_list[k]

        y = @variable(model, [1:G, 1:D], lower_bound = 0)
        u = @variable(model, [1:D], lower_bound = 0)
        v = @variable(model, [1:G], lower_bound = 0)

        @constraints(model, begin
            [g ∈ 1:G], sum(y[g, d] for d ∈ 1:D) + v[g] == x[g]
            [d ∈ 1:D], sum(y[g, d] for g ∈ 1:G) + u[d] ≥ PredictionStar[d] + y_opt[d]
            cost_recourse[k] == sum(cost[g, d]*y[g, d] for g ∈ 1:G, d ∈ 1:D) + deficit*sum(u[d] for d ∈ 1:D) + storage*sum(v[g] for g ∈ 1:G)
        end)

    end

    @objective(model, Min, (1/m)*sum(cost_recourse[k] for k in 1:m))
    optimize!(model)
    status = termination_status(model)
    if status != MOI.OPTIMAL
        error(" Status: $(status)")
    end

    x = value.(model[:x])

    return x
   
end

function dual_vertices(data, instalations, demand, G, D, x, y_opt, PredictionStar)

    model = recourse(data, instalations, demand, x, G, D)
    m_oracle = dualize(model; dual_names = DualNames())
    set_optimizer(m_oracle, optimizer_with_attributes(() -> Gurobi.Optimizer(GUROBI_ENV), "OutputFlag" => 0))

    ex1 = @expression(m_oracle, QuadExpr())
    ex2 = @expression(m_oracle, AffExpr())
    ex3 = @expression(m_oracle, AffExpr())

    for d in 1:D

        π = m_oracle[Symbol("π[$d]")]

        add_to_expression!(ex3, PredictionStar[d], π)
        add_to_expression!(ex3, y_opt[d], π)

    end

    @objective(m_oracle, Max, obj + ex1 + ex2 + ex3)
    optimize!(m_oracle)
    status = termination_status(m_oracle)
    if status != MOI.OPTIMAL
        error(" Status: $(status)")
    end

    π_opt = [value(m_oracle[Symbol("π[$d]")]) for d ∈ 1:D]
    σ_opt = [value(m_oracle[Symbol("σ[$g]")]) for g ∈ 1:G]

    return π_opt, σ_opt
end

function initilization_restricted_master_Benders(data, instalations, demand, G, D, m_master, ξ_dict, Ξ_y, zstar, PredictionStar, α)

    N = data["sample_size"]
    z = ξ_dict["z"]
    y_scen = ξ_dict["y"]
    a = Ξ_y[:,1]
    b = Ξ_y[:,2]

    m = ceil(Int, N*α)

    dist = zeros(N)

    for n in 1:N

        dz = norm(z[n] - zstar, 1)

        dy = 0.0
        for d in 1:D
            if y_scen[n][d] < a[d]
                dy += a[d] - y_scen[n][d]
            elseif y_scen[n][d] > b[d]
                dy += y_scen[n][d] - b[d]
            end
        end

        dist[n] = dz + dy

    end

    idx = sortperm(dist)

    y_list = []

    for k in 1:m

        n = idx[k]

        yproj = clamp.(y_scen[n], a, b)
        push!(y_list, yproj)

    end

    x = deterministic_equivalent_problem(data, instalations, demand, G, D, y_list, PredictionStar)

    for k ∈ 1:m 

        n = idx[k]

        yproj = y_list[k]

        π_opt, σ_opt = dual_vertices(data, instalations, demand, G, D, x, yproj, PredictionStar)
        norm_l1_z = norm(zstar - z[n], 1)
        norm_l1_y = norm(yproj - y_scen[n], 1)

        @constraint(m_master, sum(σ_opt[g]*m_master[:x][g] for g ∈ 1:G) + sum(π_opt[d]*yproj[d] for d ∈ 1:D) + sum(PredictionStar[d]*π_opt[d] for d ∈ 1:D) - m_master[:λ]*norm_l1_y - m_master[:λ]*norm_l1_z ≤ m_master[:μ][n] + m_master[:θ])

    end

    return nothing
 
end

function warm_start_solution_CCG(data, instalations, demand, G, D, ξ_dict, Ξ_y_new, y_old, L_old, zstar_new, PredictionStar_new, ρ, α)

    m_master = create_master_problem(data, G, ρ)

    N = data["sample_size"]

    deficit = data["fixed_cost"]["deficit"]
    storage = data["fixed_cost"]["storage"]
    cost = [norm([instalations["$G"][g], demand["$D"][d]], 2) for g ∈ 1:G, d ∈ 1:D]

    y_scen = ξ_dict["y"]
    z = ξ_dict["z"]

    a = Ξ_y_new[:,1]
    b = Ξ_y_new[:,2]

    m = ceil(Int, N*α)

    dist = zeros(N)

    for n in 1:N

        dz = norm(z[n] - zstar_new, 1)

        dy = 0.0
        for d in 1:D
            if y_scen[n][d] < a[d]
                dy += a[d] - y_scen[n][d]
            elseif y_scen[n][d] > b[d]
                dy += y_scen[n][d] - b[d]
            end
        end

        dist[n] = dz + dy

    end

    idx = sortperm(dist)

    for k ∈ 1:m

        n = idx[k]

        yproj = [clamp(y_scen[n][d], a[d], b[d]) for d ∈ 1:D]

        norm_l1_z = norm(zstar_new - z[n], 1)
        norm_l1_y = norm(yproj - y_scen[n], 1)

        y = @variable(m_master, [1:G, 1:D], lower_bound = 0)
        u = @variable(m_master, [1:D], lower_bound = 0)
        v = @variable(m_master, [1:G], lower_bound = 0)
        @constraints(m_master, begin
                    [g ∈ 1:G], sum(y[g, d] for d ∈ 1:D) + v[g] == m_master[:x][g]
                    [d ∈ 1:D], sum(y[g, d] for g ∈ 1:G) + u[d] ≥ PredictionStar_new[d] + yproj[d]
        end)
        @constraint(m_master, sum(cost[g, d]*y[g, d] for g ∈ 1:G, d ∈ 1:D) + deficit*sum(u[d] for d ∈ 1:D) + storage*sum(v[g] for g ∈ 1:G) - m_master[:λ]*norm_l1_y - m_master[:λ]*norm_l1_z ≤ m_master[:μ][n] + m_master[:θ])

    end

    for n ∈ collect(1:N)

        if ~isempty(L_old["$n"])

            y_opt = y_old["$n"]["best_vertice"]
            norm_l1_z = norm(zstar_new - z[n], 1)
            norm_l1_y = norm(y_opt - y_scen[n], 1)

            y = @variable(m_master, [1:G, 1:D], lower_bound = 0)
            u = @variable(m_master, [1:D], lower_bound = 0)
            v = @variable(m_master, [1:G], lower_bound = 0)
            @constraints(m_master, begin
                        [g ∈ 1:G], sum(y[g, d] for d ∈ 1:D) + v[g] == m_master[:x][g]
                        [d ∈ 1:D], sum(y[g, d] for g ∈ 1:G) + u[d] ≥ PredictionStar_new[d] + y_opt[d]
            end)
            @constraint(m_master, sum(cost[g, d]*y[g, d] for g ∈ 1:G, d ∈ 1:D) + deficit*sum(u[d] for d ∈ 1:D) + storage*sum(v[g] for g ∈ 1:G) - m_master[:λ]*norm_l1_y - m_master[:λ]*norm_l1_z ≤ m_master[:μ][n] + m_master[:θ])

        end

    end

    optimize!(m_master)
    status = termination_status(m_master)
    if status != MOI.OPTIMAL
        error("problem $status")
    end

    x_warm = value.(m_master[:x])
    λ_warm= value.(m_master[:λ])

    μ_warm = value.(m_master[:μ])
    θ_warm = value(m_master[:θ])

    return x_warm, λ_warm, μ_warm, θ_warm
 
end

function warm_start_solution_Benders(data, instalations, demand, G, D, ξ_dict, Ξ_y_new, y_old, L_old, zstar_new, PredictionStar_new, ρ_new, α)

    m_master = create_master_problem(data, G, ρ_new)

    N = data["sample_size"]

    y_scen = ξ_dict["y"]
    z = ξ_dict["z"]

    a = Ξ_y_new[:,1]
    b = Ξ_y_new[:,2]

    m = ceil(Int, N*α)

    dist = zeros(N)

    for n in 1:N

        dz = norm(z[n] - zstar_new, 1)

        dy = 0.0
        for d in 1:D
            if y_scen[n][d] < a[d]
                dy += a[d] - y_scen[n][d]
            elseif y_scen[n][d] > b[d]
                dy += y_scen[n][d] - b[d]
            end
        end

        dist[n] = dz + dy

    end

    idx = sortperm(dist)
    y_list = []

    for n ∈ 1:N 

        # n = idx[k]

        yproj = [clamp(y_scen[n][d], a[d], b[d]) for d ∈ 1:D]

        push!(y_list, yproj)

    end

    x = deterministic_equivalent_problem(data, instalations, demand, G, D, y_list, PredictionStar_new)

    for n ∈ collect(1:N)

        # n = idx[k]

        yproj = y_list[n]

        π_opt, σ_opt = dual_vertices(data, instalations, demand, G, D, x, yproj, PredictionStar_new)

        norm_l1_z = norm(zstar_new - z[n], 1)
        norm_l1_y = norm(yproj - y_scen[n], 1)

        @constraint(m_master, sum(σ_opt[g]*m_master[:x][g] for g ∈ 1:G) + sum(π_opt[d]*yproj[d] for d ∈ 1:D) + sum(PredictionStar_new[d]*π_opt[d] for d ∈ 1:D) - m_master[:λ]*norm_l1_y - m_master[:λ]*norm_l1_z ≤ m_master[:μ][n] + m_master[:θ])

    end

    y_list = [Vector(y_old[n]["best_vertice"])[:] for n ∈ keys(y_old)]
    x = deterministic_equivalent_problem(data, instalations, demand, G, D, y_list, PredictionStar_new)

    for n ∈ 1:N

        if ~isempty(L_old["$n"])
            y_opt = y_old["$n"]["best_vertice"]
            π_opt = y_old["$n"]["best_π"]
            σ_opt = y_old["$n"]["best_σ"]
            # π_opt, σ_opt = dual_vertices(data, instalations, demand, x, y_opt, PredictionStar_new)
            norm_l1_z = norm(zstar_new - z[n], 1)
            norm_l1_y = norm(y_opt - y_scen[n], 1)

            @constraint(m_master, sum(σ_opt[g]*m_master[:x][g] for g ∈ 1:G) + sum(π_opt[d]*y_opt[d] for d ∈ 1:D) + sum(PredictionStar_new[d]*π_opt[d] for d ∈ 1:D) - m_master[:λ]*norm_l1_y - m_master[:λ]*norm_l1_z ≤ m_master[:μ][n] + m_master[:θ])
        end

    end

    optimize!(m_master)
    status = termination_status(m_master)
    if status != MOI.OPTIMAL
        error("problem $status")
    end

    x_warm = value.(m_master[:x])
    λ_warm= value.(m_master[:λ])

    μ_warm = value.(m_master[:μ])
    θ_warm = value(m_master[:θ])

    return x_warm, λ_warm, μ_warm, θ_warm
 
end

function solve_master_problem(m_master)

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

function critical_radius(data, ξ_dict, Ξ_y, α, zstar; p = 1)

    N = data["sample_size"]

    z = ξ_dict["z"]
    y = ξ_dict["y"]

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

G, D = 20, 50

sample = collect(1:data["sample_size"])

# Context

p = 3 # Dimension of the context
zstar = trunc.(rand(Normal(1,0.5), p), digits=2)

ξ_dict = Dict("z" => [zeros(p) for n ∈ sample], "y" => [zeros(D) for n ∈ sample], "Ξ_y(z^*)" => zeros(D, 2))

Models = scenarios(data, D, ξ_dict, p)

Ξ_y = uncertainty_set(ξ_dict, zstar)

PredictionStar = forecast(D, zstar, Models)

# Trimming parameters

α = 0.5
ρ = critical_radius(data, ξ_dict, Ξ_y, α, zstar) 

# CCG Algorithm

violation = Dict("$n" => [] for n ∈ sample)
y_dict = Dict("$n" => [] for n ∈ sample)
y_old_CCG = Dict("$n" => Dict("best_score" => -Inf, "best_violation" => 0.0, "best_vertice" => zeros(D), "best_π" => zeros(D), "best_σ" => zeros(G)) for n ∈ sample)
π_dict = Dict("$n" => [] for n ∈ sample) 
σ_dict = Dict("$n" => [] for n ∈ sample)
L_CCG = Dict("$n" => [] for n ∈ sample) 
tol = [true for n ∈ sample]
iter = 1
m_master = create_master_problem(data, G, ρ)
initilization_restricted_master_CCG(data, instalations, demand, G, D, m_master, ξ_dict, Ξ_y, zstar, PredictionStar, α)
m_oracle, obj  = create_oracle_problem(data, instalations, demand, G, D)
while Base.any(tol) && iter ≤ 20
    global x_CCG, λ_CCG, μ_CCG, θ_CCG = solve_master_problem(m_master)
    solve_oracle_problem(data, m_oracle, obj, ξ_dict, Ξ_y, x_CCG, λ_CCG, μ_CCG, θ_CCG, G, D, tol, y_dict, y_old_CCG, π_dict, σ_dict, L_CCG, violation, zstar, PredictionStar)
    update_master_CCG(data, m_master, ξ_dict, y_dict, G, D, tol, zstar, PredictionStar)
    iter +=  1
end

# Benders Multi-Cut Algorithm

violation = Dict("$n" => [] for n ∈ sample)
y_dict = Dict("$n" => [] for n ∈ sample) 
y_old_Benders = Dict("$n" => Dict("best_score" => -Inf, "best_violation" => 0.0, "best_vertice" => zeros(D), "best_π" => zeros(D), "best_σ" => zeros(G)) for n ∈ sample)
π_dict = Dict("$n" => [] for n ∈ sample) 
σ_dict = Dict("$n" => [] for n ∈ sample)
L_Benders = Dict("$n" => [] for n ∈ sample) 
tol = [true for n ∈ sample]
iter = 1
m_master = create_master_problem(data, G, ρ)
initilization_restricted_master_Benders(data, instalations, demand, G, D, m_master, ξ_dict, Ξ_y, zstar, PredictionStar, α)
m_oracle, obj  = create_oracle_problem(data, instalations, demand, G, D)
while Base.any(tol) && iter ≤ 100
    global x_Benders, λ_Benders, μ_Benders, θ_Benders = solve_master_problem(m_master)
    solve_oracle_problem(data, m_oracle, obj, ξ_dict, Ξ_y, x_Benders, λ_Benders, μ_Benders, θ_Benders, G, D, tol, y_dict, y_old_Benders, π_dict, σ_dict, L_Benders, violation, zstar, PredictionStar)
    update_master_Benders(data, m_master, ξ_dict, y_dict, π_dict, σ_dict, G, D, tol, zstar, PredictionStar)
    iter +=  1
end

# New context
ϵ = 2
zstar_new = trunc.(zstar .+ (rand(length(zstar)) .- 0.5) .* (2 * ϵ), digits=2)

Ξ_y_new = uncertainty_set(ξ_dict, zstar_new)

PredictionStar_new = forecast(D, zstar_new, Models)
ρ_new = critical_radius(data, ξ_dict, Ξ_y_new, α, zstar_new) 

# CCG Algorithm new context

violation = Dict("$n" => [] for n ∈ sample)
y_dict = Dict("$n" => [] for n ∈ sample)
y_old_CCG_new = Dict("$n" => Dict("best_score" => -Inf, "best_violation" => 0.0, "best_vertice" => zeros(D), "best_π" => zeros(D), "best_σ" => zeros(G)) for n ∈ sample)
σ_dict = Dict("$n" => [] for n ∈ sample)
L_CCG_new = Dict("$n" => [] for n ∈ sample) 
tol = [true for n ∈ sample]
iter = 1
m_master = create_master_problem(data, G, ρ_new)
initilization_restricted_master_CCG(data, instalations, demand, G, D, m_master, ξ_dict, Ξ_y_new, zstar_new, PredictionStar_new, α)
m_oracle, obj  = create_oracle_problem(data, instalations, demand, G, D)
while Base.any(tol) && iter ≤ 20
    global x_CCG_new, λ_CCG_new, μ_CCG_new, θ_CCG_new = solve_master_problem(m_master)
    solve_oracle_problem(data, m_oracle, obj, ξ_dict, Ξ_y_new, x_CCG_new, λ_CCG_new, μ_CCG_new, θ_CCG_new, G, D, tol, y_dict, y_old_CCG_new, π_dict, σ_dict, L_CCG_new, violation, zstar_new, PredictionStar_new)
    update_master_CCG(data, m_master, ξ_dict, y_dict, G, D, tol, zstar_new, PredictionStar_new)
    iter +=  1
end

# CCG with warm start

violation = Dict("$n" => [] for n ∈ sample)
y_dict = Dict("$n" => [] for n ∈ sample)
y_old_CCG_new_with_warm_start = Dict("$n" => Dict("best_score" => -Inf, "best_violation" => 0.0, "best_vertice" => zeros(D), "best_π" => zeros(D), "best_σ" => zeros(G)) for n ∈ sample)
π_dict = Dict("$n" => [] for n ∈ sample) 
σ_dict = Dict("$n" => [] for n ∈ sample)
L_CCG_new_with_warm_start = Dict("$n" => [] for n ∈ sample) 
tol = [true for n ∈ sample]
iter = 1
m_master = create_master_problem(data, G, ρ_new)
initilization_restricted_master_CCG(data, instalations, demand, G, D, m_master, ξ_dict, Ξ_y_new, zstar_new, PredictionStar_new, α)
m_oracle, obj  = create_oracle_problem(data, instalations, demand, G, D)
while Base.any(tol) && iter ≤ 20
    if iter == 1
        global x_CCG_new_with_warm_start, λ_CCG_new_with_warm_start, μ_CCG_new_with_warm_start, θ_CCG_new_with_warm_start = warm_start_solution_CCG(data, instalations, demand, G, D, ξ_dict, Ξ_y_new, y_old_CCG, L_CCG, zstar_new, PredictionStar_new, ρ_new, α)
    else
        global x_CCG_new_with_warm_start, λ_CCG_new_with_warm_start, μ_CCG_new_with_warm_start, θ_new_with_warm_start = solve_master_problem(m_master)
    end
    solve_oracle_problem(data, m_oracle, obj, ξ_dict, Ξ_y_new, x_CCG_new_with_warm_start, λ_CCG_new_with_warm_start, μ_CCG_new_with_warm_start, θ_CCG_new_with_warm_start, G, D, tol, y_dict, y_old_CCG_new_with_warm_start, π_dict, σ_dict, L_CCG_new, violation, zstar_new, PredictionStar_new)
    update_master_CCG(data, m_master, ξ_dict, y_dict, G, D, tol, zstar_new, PredictionStar_new)
    iter +=  1
end

# Benders Multi-Cut new context

violation = Dict("$n" => [] for n ∈ sample)
y_dict = Dict("$n" => [] for n ∈ sample) 
y_old_Benders_new = Dict("$n" => Dict("best_score" => -Inf, "best_violation" => 0.0, "best_vertice" => zeros(D), "best_π" => zeros(D), "best_σ" => zeros(G)) for n ∈ sample)
σ_dict = Dict("$n" => [] for n ∈ sample)
L_Benders_new = Dict("$n" => [] for n ∈ sample) 
tol = [true for n ∈ sample]
iter = 1
m_master = create_master_problem(data, G, ρ_new)
initilization_restricted_master_Benders(data, instalations, demand, G, D, m_master, ξ_dict, Ξ_y_new, zstar_new, PredictionStar_new, α)
m_oracle, obj  = create_oracle_problem(data, instalations, demand, G, D)
while Base.any(tol) && iter ≤ 100
    global x_Benders_new, λ_Benders_new, μ_Benders_new, θ_Benders_new = solve_master_problem(m_master)
    solve_oracle_problem(data, m_oracle, obj, ξ_dict, Ξ_y_new, x_Benders_new, λ_Benders_new, μ_Benders_new, θ_Benders_new, G, D, tol, y_dict, y_old_Benders_new, π_dict, σ_dict, L_Benders_new, violation, zstar_new, PredictionStar_new)
    update_master_Benders(data, m_master, ξ_dict, y_dict, π_dict, σ_dict, G, D, tol, zstar_new, PredictionStar_new)
    iter +=  1
end

# Benders Multi-Cut with warm start

violation = Dict("$n" => [] for n ∈ sample)
y_dict = Dict("$n" => [] for n ∈ sample) 
y_old_Benders_new_with_warm_start = Dict("$n" => Dict("best_score" => -Inf, "best_violation" => 0.0, "best_vertice" => zeros(D), "best_π" => zeros(D), "best_σ" => zeros(G)) for n ∈ sample)
π_dict = Dict("$n" => [] for n ∈ sample) 
σ_dict = Dict("$n" => [] for n ∈ sample)
L_Benders_new_with_warm_start = Dict("$n" => [] for n ∈ sample) 
tol = [true for n ∈ sample]
iter = 1
m_master = create_master_problem(data, G, ρ_new)
initilization_restricted_master_Benders(data, instalations, demand, G, D, m_master, ξ_dict, Ξ_y_new, zstar_new, PredictionStar_new, α)
m_oracle, obj  = create_oracle_problem(data, instalations, demand, G, D)
while Base.any(tol) && iter ≤ 100
        if iter == 1
            global x_Benders_new_with_warm_start, λ_Benders_new_with_warm_start, μ_Benders_new_with_warm_start, θ_Benders_new_with_warm_start  = warm_start_solution_Benders(data, instalations, demand, G, D, ξ_dict, Ξ_y_new, y_old_CCG, L_CCG, zstar_new, PredictionStar_new, ρ_new, α)
        else
            global x_Benders_new_with_warm_start, λ_Benders_new_with_warm_start, μ_Benders_new_with_warm_start, θ_Benders_new_with_warm_start = solve_master_problem(m_master)
        end
    solve_oracle_problem(data, m_oracle, obj, ξ_dict, Ξ_y, x_Benders_new_with_warm_start, λ_Benders_new_with_warm_start, μ_Benders_new_with_warm_start, θ_Benders_new_with_warm_start, G, D, tol, y_dict, y_old_Benders_new_with_warm_start, π_dict, σ_dict, L_Benders_new_with_warm_start, violation, zstar_new, PredictionStar_new)
    update_master_Benders(data, m_master, ξ_dict, y_dict, π_dict, σ_dict, G, D, tol, zstar_new, PredictionStar_new)
    iter +=  1
end


