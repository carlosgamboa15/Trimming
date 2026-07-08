using Distributions, LinearAlgebra, GLM, DataFrames, Statistics, Random, NearestNeighbors, StatsPlots

Random.seed!(2025)

N = 1000         # historical observations
D = 20          # demand locations
p = 3           # context variables
sample = collect(1:N)

# contexts
z = [rand(p) for n ∈ sample]

β0 = rand(D)
β = [rand(p) for d ∈ collect(1:D)]

σ = 0.30

ξ = [zeros(D) for n ∈ sample]

for d ∈ collect(1:D)
    for n ∈ sample

        μ = β0[d] + dot(β[d],z[n])  

        ξ[n][d] = rand(LogNormal(μ,σ))

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
        y = log.([ξ[n][d] for n ∈ sample]),
    )

    model = lm(@formula(y ~ z1 + z2 + z3),df)

    Models[d] = model
    Forcast = predict(model)
    for n ∈ sample
        ξ_forcast[n][d] = exp(Forcast[n])
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

PredictionStarLog = zeros(D)

for d in 1:D

    PredictionStarLog[d] = predict(
        Models[d],
        DataFrame(
            z1=[zstar[1]],
            z2=[zstar[2]],
            z3=[zstar[3]]
        )
    )[1]

end

ForecastStar = exp.(PredictionStarLog)
