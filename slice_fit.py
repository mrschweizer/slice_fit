import numpy as np
from scipy.optimize import curve_fit
import xarray as xr
import matplotlib.pyplot as plt


def iter_columns(array: np.ndarray, axis=-1, func: callable = None, **kwargs) -> np.ndarray:
    """
    Reduce nd-data to (n-1)d data. By performing the operation on one 1D axis.
    :param array: nd input data
    :param axis: 1d xis to perform reduction over
    :param func: function to be applied for reduction
    :return: Array of reduced data
    """
    # Initialize the reduced array, that is to be generated.
    reduced_shape = list(array.shape)
    reduced_shape.pop(axis)
    a = np.empty(tuple(reduced_shape))
    # Iterate over the output array
    with np.nditer(a, flags=['multi_index'], op_flags=[['writeonly']]) as it:
        for a_i in it:
            # modify multi index to slice over dimension of axis, append if axis
            mod = list(it.multi_index)
            mod.insert(axis if axis >= 0 else len(it.multi_index)+1, slice(None))
            # take the slice from the input array, perform the function on it
            # and write the result to the output destination
            a_i[...] = func(array[tuple(mod)], **kwargs)
    return a

b = np.arange(10).reshape(5, 2)
print(iter_columns(b, axis=1, func=np.sum))

def reduce_fit_wrapper(x, axis=-1, **kwargs):
    """
    Wrapping function for iter_columns and exponential fit function.
    :param x: nd input array
    :param axis: axis over which should be reduces
    :param kwargs: Remaining keyword arguments. Passed on from reduce.
    :return: scalar of resulting fit parameter
    """
    return iter_columns(x, axis=axis, func=fitting, **kwargs)



def exponential_func(x: np.ndarray, a, decrement):
    return a*np.exp(x*decrement)


def lin_fit(x: np.ndarray, offset, decrement):
    return x*decrement+offset


def noisify(x: np.ndarray, span: float):
    # Generate noise
    rng = np.random.default_rng()
    return (rng.random(x.shape) * span + 1 - span/2)*x


def log_fit_wrapper(f, x, y, *args, **kwargs):
    fit_parameters = curve_fit(f, x, np.log(y), *args, **kwargs)[0]
    fit_parameters[0] = np.exp(fit_parameters[0])
    return tuple(fit_parameters)


def fitting(y, *args, base=None, **kwargs):
    # plt.plot(base, y)
    fit_parameters = curve_fit(lin_fit, base, np.log(y), *args, check_finite=False, **kwargs)[0]
    # fit_parameters[0] = np.exp(fit_parameters[0])
    print(fit_parameters[0])
    return fit_parameters[1]


# Generate test data
time = np.linspace(0, 1, 201)
exp = np.linspace(0, 1, 5)
t, s = np.meshgrid(time, exp)
data = np.exp(t*s)
del(t, s)


# Plot data
fig, ax = plt.subplots()
arr = xr.DataArray(data, dims=('exp', 'time'), coords={'time': time, 'exp': exp}) .reduce(reduce_fit_wrapper, dim='time', base=time)
arr.plot.step(ax=ax,where='mid')

"""
params_noise = noisify(np.array(list(params)), 0.4)
fit_param = log_fit_wrapper(lin_fit, time, ydata, p0=params_noise, )
fit_data = exponential_func(time, *fit_param)
"""



plt.show()
