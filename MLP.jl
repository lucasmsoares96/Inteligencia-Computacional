### A Pluto.jl notebook ###
# v0.19.15

using Markdown
using InteractiveUtils

# ╔═╡ 2070ca57-d02e-492e-afaf-081dc4ca6559
begin
	import Pkg
	Pkg.activate()
	using Plots
	using PlutoUI
	using Statistics
	using Random
	TableOfContents(title="Índice")
end

# ╔═╡ 1beb82cf-43a5-4fab-b311-7c096b435413
md"""
# Multilayer Perceptron
"""

# ╔═╡ e5d1d090-1bd5-4e3f-b39c-e324d525a47f
md"""
## Introdução
Neste relatório será apresentado uma implementação e uma análiase do algoritmo backpropagation do multilayer perceptron aplicado ao problema XOR (ou exclusivo). Este problema é um exemplo simples de uma solução não linearmente separável, no qual uma única camada não consegue resolver. Em resumo o multilayer perceptron consiste em encadear todas as saidas de uma camada a todas as entradas da camada seguinte, retro propagar o erro e ajustar os pesos da última camada para a primeira camada.
"""

# ╔═╡ b2b141c5-e211-44d2-912a-a0d50805b004
LocalResource("maxresdefault.jpg")

# ╔═╡ 71442437-e06c-4d6a-a37a-0f977429ea0f
md"""
Para a realização desse trabalho foram utilizados como referência:
1. Slides da disciplina
1. Playlist do canal [3Blue1Brown](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
1. Playlist do canal [The Code Train](https://www.youtube.com/playlist?list=PLRqwX-V7Uu6aCibgK1PTWWu9by6XFdCfh)
1. Livro [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)

Esse material foi escolhido por apresentar uma implementação simples de uma rede MLP utilizando a notação de matriz, o que facilita o entendimento dos conceitos
"""

# ╔═╡ a8df7778-538b-4bc9-8646-7a5393a6645a
md"""
## Implementação
Primeiramente precisamos definir o nosso vetor de entrada x e o nosso vetor de saída y para o problema XOR.
"""

# ╔═╡ 7d9649e6-3d75-4889-a9e2-4fce6ef669be
x = [
	0 0
	0 1
	1 0
	1 1
]

# ╔═╡ f2b5359c-510a-495b-a338-2337c2eb3136
y = [
	0
	1
	1
	0
]

# ╔═╡ 254dfe10-6816-4e62-bff7-f78d63fe99f0
md"""
Em seguida devemos definir a nossa função de ativação e a derivada da função de ativação que será utilizada no cálculo do vetor gradiente.
"""

# ╔═╡ e1d0412f-3e30-4fbd-b9a7-27e53e2d0521
σ(x) = 1 / ( 1 + exp(-x) )

# ╔═╡ 808804a5-7c81-4f9c-b6e8-090f293d9b16
dσ(x) =  x * (1 - x)

# ╔═╡ dc634efa-27e6-454b-81e3-1fc1e3ec392a
md"""
Também precisamos definir uma estrutura que armazenará as informações da nossa rede neural.
"""

# ╔═╡ 2fb6bf23-2586-4d49-ac5e-1884a18981ca
# outer constructor
# ou com Base.@kwdef
begin
	mutable struct NeuralNetwork
		input_nodes
		hidden_nodes
		output_nodes
		weights_ih
		weights_ho
		bias_h
		bias_o
		η
		last_e
	end
	NeuralNetwork(i,h,o,η) = NeuralNetwork(
		i,
		h,
		o,
		rand(-1:0.01:1,h,i),
		rand(-1:0.01:1,o,h),
		rand(-1:0.01:1,h,1),
		rand(-1:0.01:1,o,1),
		η,
		0
	)
end

# ╔═╡ ef5367b4-811a-4b4a-8bdb-3f10c22351b8
md"""
Com isso temos o necessário para implementar as funções de treinamento da rede e a função de previsão.
"""

# ╔═╡ 36462e38-b96b-4a9d-9caa-2c51401c8a0e
md"""
A função de previsão recebe como argumento uma rede neural e uma entrada para prever a saída. A saída da camada oculta é obtida multiplicando a matriz de pesos da camada interna para a oculta pela a matriz entrada somada com a matriz de bias da camada oculta. Já a saída da camada de saída é obtida multiplicando a matriz dos pesos entre a camada oculta e a camada de saída pela matriz de saída da camada oculta somada com a matriz de bias da camada de saída.
"""

# ╔═╡ 5b4ec109-f988-4bf1-92cc-e2e40f4dd0c4
begin
	predict(a) = x -> predict(a,x)
	function predict(this::NeuralNetwork,X)
		H = this.weights_ih * X .+ this.bias_h .|> σ
		O = this.weights_ho * H .+ this.bias_o .|> σ
		O .|> round
	end
end

# ╔═╡ d02b825a-70aa-4549-9d16-606994f39de4
md"""
A função de treinamento recebe como argumento uma rede neural, uma amostra dos dados de entrada e a saída desejada. O algoritmo consiste em 3 passos: Feedforward, Backpropagation Output e Backpropagation Hidden. A etapa Feedforward consiste na previsão do sistema. Em seguida, a saída desejada será utilizada para calcular o erro da rede, subtraíndo a pela saída da rede. A partir deste erro podemos ajustar o peso para melhorar a precisão da rede nas etapas de Backpropagation Output e Hidden.
"""

# ╔═╡ 0aa61377-a374-4c5e-9aa2-54fede251cc4
begin
	fit!(X,Y) = nn -> fit(nn,X,Y)
	function fit!(this::NeuralNetwork,X,Y)
		
		### Feedforward
		
		H = σ.(this.weights_ih * X + this.bias_h)
		O = σ.(this.weights_ho * H + this.bias_o)
	
		### Backpropagation Output
		
		Oe = Y .- O
		O∇ = dσ.(O) .* Oe .* this.η
		
		δho = O∇ * H'
		this.weights_ho += δho
		this.bias_o     += O∇
	
		### Backpropagation Hidden
	
		He = this.weights_ho' * Oe
		H∇ = dσ.(H) .* He .* this.η
		δih = H∇ * X'
	
		this.weights_ih += δih
		this.bias_h     += H∇
		
		this.last_e += mean(abs.(Oe))
	end
end

# ╔═╡ fbd1066f-eb42-49cf-908e-1b93b281f6f9
md"""
Por fim devemos treinar o nosso modelo para todas as entradas de X pela quantidade de épocas estabelecidas até se atingit um erro dentro da tolerância.
"""

# ╔═╡ aa075c80-a5b6-4e9d-87e1-69ccfee56792
begin
	push!(y) = x -> Base.push!(y,x)
	nn = NeuralNetwork(
		2,
		2,
		1,
		0.3
	)
	τ = 0.05
	Ω = 10000
	epocas = 0
	erroEpoca = Float64[]
	
	while true
		nn.last_e = 0
		
		size(x,1) |> randperm .|> i -> fit!(nn,x[i,:],y[i])
		
		# for i in randperm(size(x,1))
		# 	fit!(nn,x[i,:],y[i])
		# end
		
		global epocas += 1
		nn.last_e / length(x) |> push!(erroEpoca)
		
		(epocas ≥ Ω || erroEpoca[end] ≤ τ) ? break : continue
	end
end

# ╔═╡ 537976fd-1fb7-4235-8759-dfe53deeb9f5
md"""
## Análise
"""

# ╔═╡ 10436ba9-4bca-4a02-8d8d-48a44550c8b7
md"""
É possível notar que o a rede demora certa de 1000 iterações para atingir um erro aceitável.
"""

# ╔═╡ 30c1d9da-5ac2-4217-8669-16871cafa9a0
println("Epocas: ", epocas, " \n", "Erro:   ", erroEpoca[end])

# ╔═╡ fb498595-7ca6-4af5-ad86-85c1a63f753e
plot(1:epocas, erroEpoca)

# ╔═╡ fc1c4258-5ce2-4400-b3c8-72cecaa80a8a
nn

# ╔═╡ 1f7d1208-2c25-4fb0-b7a6-da55bfd2b7ba
x |> eachrow .|> predict(nn) .|> println;

# ╔═╡ Cell order:
# ╠═2070ca57-d02e-492e-afaf-081dc4ca6559
# ╟─1beb82cf-43a5-4fab-b311-7c096b435413
# ╟─e5d1d090-1bd5-4e3f-b39c-e324d525a47f
# ╟─b2b141c5-e211-44d2-912a-a0d50805b004
# ╟─71442437-e06c-4d6a-a37a-0f977429ea0f
# ╟─a8df7778-538b-4bc9-8646-7a5393a6645a
# ╠═7d9649e6-3d75-4889-a9e2-4fce6ef669be
# ╠═f2b5359c-510a-495b-a338-2337c2eb3136
# ╟─254dfe10-6816-4e62-bff7-f78d63fe99f0
# ╠═e1d0412f-3e30-4fbd-b9a7-27e53e2d0521
# ╠═808804a5-7c81-4f9c-b6e8-090f293d9b16
# ╟─dc634efa-27e6-454b-81e3-1fc1e3ec392a
# ╠═2fb6bf23-2586-4d49-ac5e-1884a18981ca
# ╟─ef5367b4-811a-4b4a-8bdb-3f10c22351b8
# ╟─36462e38-b96b-4a9d-9caa-2c51401c8a0e
# ╠═5b4ec109-f988-4bf1-92cc-e2e40f4dd0c4
# ╟─d02b825a-70aa-4549-9d16-606994f39de4
# ╠═0aa61377-a374-4c5e-9aa2-54fede251cc4
# ╟─fbd1066f-eb42-49cf-908e-1b93b281f6f9
# ╠═aa075c80-a5b6-4e9d-87e1-69ccfee56792
# ╟─537976fd-1fb7-4235-8759-dfe53deeb9f5
# ╟─10436ba9-4bca-4a02-8d8d-48a44550c8b7
# ╠═30c1d9da-5ac2-4217-8669-16871cafa9a0
# ╠═fb498595-7ca6-4af5-ad86-85c1a63f753e
# ╠═fc1c4258-5ce2-4400-b3c8-72cecaa80a8a
# ╠═1f7d1208-2c25-4fb0-b7a6-da55bfd2b7ba
