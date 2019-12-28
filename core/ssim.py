import numpy as np

__all__ = ('Thumbnail', 'Thumbnail_3', 'compare_ssim')

_integer_types = (np.byte, np.ubyte,  # 8 bits
                  np.short, np.ushort,  # 16 bits
                  np.intc, np.uintc,  # 16 or 32 or 64 bits
                  np.int_, np.uint,  # 32 or 64 bits
                  np.longlong, np.ulonglong)  # 64 bits
_integer_ranges = {t: (np.iinfo(t).min, np.iinfo(t).max)
                   for t in _integer_types}
dtype_range = {np.bool_: (False, True),
               np.bool8: (False, True),
               np.float16: (-1, 1),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}
dtype_range.update(_integer_ranges)


class Thumbnail_3:
    """
        Wrap class of 3-channel image for SSIM comparision.
        Accelerate repeated comparisons by saving intermediate calculations.
    """

    def __init__(self, input, win_size=7, initial=True):
        assert input.ndim == 3 and input.shape[2] == 3, 'Accept image with 3 channels only.'
        self.ch_0 = Thumbnail(input[..., 0], win_size, initial)
        self.ch_1 = Thumbnail(input[..., 1], win_size, initial)
        self.ch_2 = Thumbnail(input[..., 2], win_size, initial)

    def cmp_ssim(self, other):
        assert isinstance(other, Thumbnail_3)
        ch_0 = self.ch_0 - other.ch_0
        ch_1 = self.ch_1 - other.ch_1
        ch_2 = self.ch_2 - other.ch_2
        return np.average((ch_0, ch_1, ch_2))

    def __sub__(self, other):
        return self.cmp_ssim(other)

    def array(self):
        return np.dstack((self.ch_0, self.ch_1, self.ch_2))

    def initial(self):
        self.ch_0.initial()
        self.ch_1.initial()
        self.ch_2.initial()

    def reset(self):
        self.ch_0.reset()
        self.ch_1.reset()
        self.ch_2.reset()


class Thumbnail:
    """
        Wrap class of 1-channel image for SSIM comparision.
        Accelerate repeated comparisons by saving intermediate calculations.
    """

    def __init__(self, input, win_size=7, initial=True):
        self.array = np.asarray(input, dtype=np.float64)
        self.shape = input.shape
        self.win_size = win_size
        self.ux = None
        self.vx = None
        self.ndim = None
        self.dtype = None
        self.size = None
        if initial:
            self.initial()

    def initial(self):
        self.ux = uniform_filter_2d(self.array, self.win_size)
        uxx = uniform_filter_2d(self.array * self.array, self.win_size)
        self.vx = (uxx - self.ux * self.ux)
        self.ndim = self.array.ndim
        self.dtype = self.array.dtype
        self.size = self.array.size

    def reset(self):
        del self.ux
        del self.vx
        del self.ndim
        del self.dtype
        del self.size

    def cmp_ssim(self, other):
        assert self.win_size == other.win_size, 'Input should be sampled with the same win_size.'
        return compare_ssim(self, other, win_size=self.win_size)

    def __sub__(self, other):
        return self.cmp_ssim(other)


def compare_ssim(X: Thumbnail, Y: Thumbnail, win_size=None, gradient=False,
                 data_range=None,
                 full=False, **kwargs):
    """
    Adept from skimage.measure.compare_ssim
    """
    if not X.shape == Y.shape:
        raise ValueError('Input images must have the same dimensions.')

    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)
    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")
    use_sample_covariance = kwargs.pop('use_sample_covariance', True)

    if win_size is None:
        win_size = 7  # backwards compatibility

    if np.any((np.asarray(X.shape) - win_size) < 0):
        raise ValueError(
            "win_size exceeds image extent.")

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if data_range is None:
        if X.dtype != Y.dtype:
            raise ValueError("Inputs have mismatched dtype.  Setting data_range based on "
                             "X.dtype.")
        dmin, dmax = dtype_range[X.dtype.type]
        data_range = dmax - dmin

    ndim = X.ndim

    NP = win_size ** ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute (weighted) means
    ux = X.ux
    uy = Y.ux

    # compute (weighted) variances and covariances
    uxy = uniform_filter_2d(X.array * Y.array, win_size)
    vx = cov_norm * X.vx
    vy = cov_norm * Y.vx
    vxy = cov_norm * (uxy - ux * uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # compute (weighted) mean of ssim
    mssim = crop(S, pad).mean()

    if gradient:
        # The following is Eqs. 7-8 of Avanaki 2009.
        grad = uniform_filter_2d(A1 / D, win_size) * X.array
        grad += uniform_filter_2d(-S / B2, win_size) * Y.array
        grad += uniform_filter_2d((ux * (A2 - A1) - uy * (B2 - B1) * S) / D, win_size)
        grad *= (2 / X.size)

        if full:
            return mssim, grad, S
        else:
            return mssim, grad
    else:
        if full:
            return mssim, S
        else:
            return mssim


def crop(S, pad):
    return S[pad:-pad, pad:-pad]


def uniform_filter_2d(input, size):
    pad = size // 2
    padded = np.pad(np.float64(input), pad, 'symmetric')
    filter = np.full((size, size), 1 / size ** 2, dtype='float64')
    return convolve2d(padded, filter)


def convolve2d(img, kernel):
    """
    Adept from:
    https://stackoverflow.com/questions/43086557/convolve2d-just-by-using-numpy
    """
    # calc the size of the array of submatracies
    sub_shape = tuple(np.subtract(img.shape, kernel.shape) + 1)

    # alias for the function
    strd = np.lib.stride_tricks.as_strided

    # make an array of submatracies
    submatrices = strd(img, kernel.shape + sub_shape, img.strides * 2)

    # sum the submatraces and kernel
    convolved_matrix = np.einsum('ij,ijkl->kl', kernel, submatrices)

    return convolved_matrix
