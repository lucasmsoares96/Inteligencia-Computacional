module NFN
using Plots

export defFunc, plotFunc, predict

trimf(x, a, m, b) = max(min((x − a) / (m − a), (b − x) / (b − m)), 0)

function defFunc(xt, ls)
    cs = size(xt, 2)
    func = Array{Function}(undef, ls, cs)
    for c in 1:cs
        minV, maxV = extrema(xt[:, c])
        step = (maxV - minV) / (ls - 1)
        mean = [minV - step]
        for _ ∈ 1:ls+1
            a = (mean |> last) + step
            push!(mean, a)
        end
        for l ∈ 1:ls
            func[l, c] = x -> trimf(x, mean[l], mean[l+1], mean[l+2])
        end
    end
    return func
end

function μf(xt, func)
    f, x = size(func)
    mat = zeros(f, x)
    for i = 1:f, j = 1:x
        mat[i, j] = func[i, j](xt[j])
    end
    return mat
end

function plotFunc(func, t)
    plots = []
    for q in axes(func, 2)
        push!(plots, plot(t, func[:, q]))
    end
    plot(plots...)
end

function predict(xt, yt, func)
    W = zeros(size(func))
    y_list = zeros(length(yt))
    for i in axes(xt, 1)
        x = xt[i, :]
        yd = yt[i]
        μ = μf(x, func)
        # display(μ)
        y = sum(μ .* W)
        y_list[i] = y
        e = y - yd
        α = 1 / sum(μ .^ 2)
        W -= α * e .* μ
    end
    return y_list
end

end