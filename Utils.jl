module Utils
# using PyCall
using XLSX
using CSV
# using Combinatorics
using Tables
using DataFrames
using MLJ
import Base

export map, filter, simulation, readData, trainModel, tSeriesVal, tSeriesVar, selectKBest

function readData()
    xlsx = XLSX.readtable(
        "./IBOVESPA/05 MINUTOS 01-01-17 A 31-08-17.xlsx",
        "LAST_PRICEQUADRIMESTRE1e22017"
    ) |> DataFrame
    xlsx = xlsx[Not(1), Not(1)]
    xlsx = xlsx[3:6000, :]
    xlsx = replaceMissing(xlsx, "#N/A N/A")
    xlsx = xlsx[completecases(xlsx), :]
    disallowmissing!(xlsx)
    xlsx = coerceDf(xlsx, Continuous) |> Matrix
end

# function selectKBest(X, y, k)
#     # feature request
#     py"""
#     from sklearn.feature_selection import SelectKBest
#     from sklearn.feature_selection import f_regression
#     """
#     SelectKBest = py"SelectKBest(f_regression, k=5)"
#     xt = SelectKBest.fit_transform(Matrix(X), Vector(y))
#     df = DataFrame(xt, ["x1", "x2", "x3", "x4", "x5"])
# end

coerceDf(df, tipo) =
    DataFrame(
        df |> eachcol .|> a -> coerce([a...], Union{Missing,tipo}),
        names(df)
    )

map(f::Function) = l -> Base.map(f, l)

filter(f::Function) = l -> Base.filter(f, l)

function replaceMissing(df, text)
    DataFrame(
        hcat((
            df |> eachcol .|> map(x -> x == text ? missing : x)
        )...), names(df)
    )
end

function tSeriesVal(acao)
    tSeries = []
    for i in 2:20
        push!(tSeries, acao[i:end-(21-i)])
    end
    tSeries = hcat(tSeries...)
    for i in 1:18
        m = tSeries[:, 19-i:19] |> eachrow .|> mean
        tSeries = hcat(tSeries, m)
    end
    X = tSeries[:, 1:end-1]
    y = tSeries[:, end]
    return X, y
end

function tSeriesVar(acao)
    tSeries = []
    for i in 2:20
        push!(tSeries, acao[i:end-(21-i)] - acao[i-1:end-(21-i+1)])
    end
    tSeries = hcat(tSeries...)
    # for i in 1:18
    #     m = tSeries[:, 19-i:19] |> eachrow .|> mean
    #     tSeries = hcat(tSeries, m)
    # end
    X = tSeries[:, 1:end-1]
    y = tSeries[:, end]
    return X, y
end


# tSeries = DataFrame(tSeries,:auto)
# stand1 = Standardizer()
# tSeries = MLJ.transform(fit!(machine(stand1, tSeries)), tSeries)


function simulation(;initValue, predict, real)
    qtdTotal = 0
    lMao = []
    cont = 0
    for i in eachindex(predict)
        currentV = real[i]
        variation = predict[i]
        if (variation > 0 && initValue > currentV)
            cont += 1
            qtdOperation = initValue ÷ currentV   
            qtdTotal += qtdOperation
            initValue -= qtdOperation * currentV  #comprar
        elseif (variation ≤ 0 && qtdTotal >0)
            cont += 1
            initValue += qtdTotal * currentV  #vender
            qtdTotal = 0
        end
        push!(lMao, initValue + (qtdTotal * currentV))
    end
    return lMao, cont

end

end