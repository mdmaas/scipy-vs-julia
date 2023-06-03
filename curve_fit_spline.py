import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
from scipy import interpolate
from scipy.interpolate import splrep, BSpline
from scipy.stats import median_abs_deviation as mad

xdata = np.linspace(0, 2*np.pi, 35)
y = np.sin(xdata)
rng = np.random.default_rng()
y_noise = 0.2 * rng.normal(size=xdata.size)
ydata = y + y_noise


def Spline(x, *p):
    t = np.linspace(0, 2*np.pi, len(p))
    s = interpolate.CubicSpline(t, np.array(p), bc_type = "natural")
    return s(x)


# Best fit of cubic spline on given nodes
p0 = np.array([0.5]*6)
total_time = 0
for i in range(100):
    start_time = time.time()
    popt, pcov = curve_fit(Spline, xdata, ydata, p0=p0, method="lm")
    t = (time.time() - start_time)
    total_time += t

print("--- BestSpline fit took %s seconds on average ---" % str(total_time/100))



# Call smoothing splines with default smoothing parameter
total_time = 0
for i in range(100):
    start_time = time.time()
    tck_s = splrep(xdata, ydata, s=len(xdata)*mad(ydata))
    t = (time.time() - start_time)
    total_time += t
    
print("--- Dierckx took %s seconds on average ---" % str(total_time/100))


plt.plot(xdata, y, 'b-', label='True function')
plt.plot(xdata, ydata, 'o', label='data')
plt.plot(xdata, Spline(xdata, *popt), 'g--', label='Best Spline')
plt.plot(xdata, BSpline(*tck_s)(xdata), '-', label='Dierckx BSpline')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


rd, rb = 0, 0
ed, eb = 0, 0
for i in range(100):
    xdata = np.linspace(0, 2*np.pi, 35)
    y = np.sin(xdata)
    y_noise = 0.2 * rng.normal(size=xdata.size)
    ydata = y + y_noise

    residual_dierckx = np.max(np.abs(BSpline(*tck_s)(xdata)-ydata))
    residual_bestspline = np.max(np.abs(Spline(xdata, *popt) - ydata))

    rd += residual_dierckx
    rb += residual_bestspline

    err_dierckx = np.max(np.abs(BSpline(*tck_s)(xdata)-y))
    err_bestspline = np.max(np.abs(Spline(xdata, *popt) - y))
    
    ed += err_dierckx
    eb += err_bestspline


print(f'Average Max error in Dierckx {rd/100}! vs best spline {rb/100}')
print(f'Average True error in Dierckx {ed/100}! vs best spline {eb/100}')


