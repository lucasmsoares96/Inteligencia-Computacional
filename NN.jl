module NN
using MLJ
using MLJFlux

export trainModel

function trainModel(dfa, ya)
    builder = MLJFlux.@builder begin
        init = Flux.glorot_uniform(rng)
        Chain(Dense(n_in, 64, relu, init=init),
            Dense(64, 32, relu, init=init),
            Dense(32, 1, init=init))
    end
    NeuralNetworkRegressor = @load NeuralNetworkRegressor pkg = MLJFlux
    model = NeuralNetworkRegressor(
        builder=builder,
        rng=123,
        epochs=30
    )
    # Taxa de Aprendizagem
    # model.optimiser.eta = 0.001
    mach = machine(model, dfa, ya)
    MLJ.fit!(mach, verbosity=2)
    MLJ.save("mach.jlso", mach)
    return mach
end

end