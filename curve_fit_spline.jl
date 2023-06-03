using DataInterpolations
using LsqFit
using Plots
using BenchmarkTools

function Spline(x,p)
    u = LinRange(0,2π,length(p))
    poly = CubicSpline(p,u)
    return poly.(x)
end


xdata = LinRange(0, 2π, 35)
ytrue = sin.(xdata)
ydata = ytrue + 0.2 * randn(size(xdata))

p0 = ones(6)*0.5
println("Running benchmark...")
@btime fit = curve_fit(Spline, xdata, ydata, p0)

fit = curve_fit(Spline, xdata, ydata, p0)

plot(xdata, ytrue, label="True function")
scatter!(xdata,ydata,marker=:o, label="Data")
plot!(xdata,Spline(xdata,coef(fit)), label="Best Spline Fit")