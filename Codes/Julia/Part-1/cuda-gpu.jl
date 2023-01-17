using CUDA
using LinearAlgebra
using Random
N = 10000

a_d = CuArray{Float32}(undef, (N ,N))
b_d = CuArray{Float32}(undef, (N ,N))
c_d = CuArray{Float32}(undef, (N ,N))

function matmul!(A, B, C)
	CUDA.randn!(A)
	CUDA.randn!(B)
	mul!(C, A, B)
end

# Use `CUDA.@time` macro to time the GPU execution and memory usage
for i in 1:10
	CUDA.@time matmul!(a_d, b_d, c_d)
end
#=
global a_d = nothing
global b_d = nothing
global c_d = nothing

GC.gc()
=#
