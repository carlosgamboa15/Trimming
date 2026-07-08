using Distributions, LinearAlgebra, GLM, DataFrames, Statistics, Random, NearestNeighbors, StatsPlots, JSON

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

Random.seed!(2025)

N = data["sample_size"] # historical observations
p = 3                   # context variables
sample = collect(1:N)

# contexts
z = [rand(p) for n ∈ sample]
ξ_dict["$D"]["z"] = z

ξ_dict = Dict("$D" => Dict("zstar" => zeros(p), "PredictionStar" => zeros(D), "z" => [zeros(p) for n ∈ sample], "y" => [zeros(D) for n ∈ sample], "Ξ_y(z^*)" => zeros(D, 2)) for D ∈ data["demand"])

for D ∈ data["demand"]

    β0 = fill(30.0, D) 
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
    Residuals = [ε[n][d] for n ∈ sample, d ∈ collect(1:D)]

    tree = KDTree(permutedims(Z))

    zstar = randn(p)

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

    ξ_dict["$D"]["zstar"] = zstar
    ξ_dict["$D"]["PredictionStar"] = PredictionStar
    for n ∈ sample
        for d ∈ collect(1:D)
            ξ_dict["$D"]["y"][n][d] = ε[n][d]
        end
    end
    ξ_dict["$D"]["Ξ_y(z^*)"] = Matrix([Lower[:] Upper[:]])

end

save_json(ξ_dict, pwd(), "ξ_dict")
