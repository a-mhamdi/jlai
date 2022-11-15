# Fourier Series Decomposition
using Waveforms
#= https://juliapackages.com/p/waveforms =#
using LaTeXStrings
using Plots

t = range(0, 5, step=.01)
wt = 2*pi*t/1 # Frequency = 1Hz

## SQUARE WAVE
x = [ squarewave(theta) for theta in wt ]
plot(t, x, label=L"x(t)")

### 4/pi*(sin(wt)/1+sin(3wt)/3+sin(5wt)/5+...)
label_x = L"\dfrac{4}{\pi}\sum_{k\in\mathbb{N}}\dfrac{\sin((2k+1)wt)}{2k+1}"
series_x = [ 4/(k*pi)*[ sin(k*theta) for theta in wt ] for k in range(1, 20, step=2) ]
xapp = sum(series_x, dims=1)

px = plot!(t, xapp, label=label_x, title="Square Wave")

## TRIANGULAR WAVE
y = [ trianglewave(theta-pi/2) for theta in wt ]

plot(t, y, label=L"y(t)")

### -8/pi^2*(cos(wt)/1^2+cos(5wt)/5^2+...)
label_y = L"-\dfrac{8}{\pi^2}\sum_{k\in\mathbb{N}}\dfrac{\cos((2k+1)wt)}{(2k+1)^2}"
series_y = [ [ -8/((k*pi)^2)*cos(k*theta) for theta in wt ] for k=1:2:20 ]
yapp = sum(series_y, dims=1)

py = plot!(t, yapp, label=label_y, title="Triangular Wave")

## SAWTOOTH WAVE
z = [ -sawtoothwave(theta-pi) for theta in wt ]
plot(t, z, label=L"z(t)")

### 2/pi*(sin(wt)+sin(2wt)/2+sin(3wt)/3+...)
label_z = L"\dfrac{2}{\pi}\sum_{k\in\mathbb{N}^*}\dfrac{\sin(kwt)}{k}"
series_z = [ 2/(k*pi)* [sin(k*theta) for theta in wt ] for k in 1:20 ]
zapp = sum(series_z, dims=1)

pz = plot!(t, zapp, label=label_z, title="Sawtooth Wave")

graphs = plot(px, py, pz, layout=(3,1))

