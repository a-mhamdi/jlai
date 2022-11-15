using LaTeXStrings
using ControlSystems
using Plots; gr()

tspan = 0:.01:10

s = tf("s")

plot()
for τ in [.25, .5, .8, 1, 1.35]
    sys = 1/(1+τ*s)
    # PLOT THE STEP RESPONSE
    plot!(step(sys, tspan), xlabel="t (sec)", label=L"\tau"*"=$τ", legend=:bottomright)
end
current()

