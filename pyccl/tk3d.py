from . import ccllib as lib

from .pyutils import check
import numpy as np


class Tk3D(object):
    def __init__(self, a_arr, lk_arr, tkk_arr=None,
                 pk1_arr=None, pk2_arr=None, extrap_order_lok=1,
                 extrap_order_hik=1, is_logt=True):
        na = len(a_arr)
        nk = len(lk_arr)

        if ((extrap_order_hik not in (0, 1)) or
            (extrap_order_lok not in (0, 1))):
            raise ValueError("Only constant or linear extrapolation in "
                             "log(k) is possible (`extrap_order_hik` or "
                             "`extrap_order_lok` must be 0 or 1).")
        status = 0

        if tkk_arr is None:
            if pk2_arr is None:
                pk2_arr = pk1_arr
            if (pk1_arr is None) or (pk2_arr is None):
                raise ValueError("If trispectrum is factorizable "
                                 "you must provide the two factors")
            if (pk1_arr.shape != (na, nk)) or (pk2_arr.shape != (na, nk)):
                raise ValueError("Input trispectrum factor "
                                 "shapes are wrong")

            self.tsp, status = lib.set_tk3d_new_factorizable(lk_arr, a_arr,
                                                             pk1_arr.flatten(),
                                                             pk2_arr.flatten(),
                                                             int(extrap_order_lok),
                                                             int(extrap_order_lok),
                                                             int(is_logt), status)
        else:
            if tkk_arr.shape != (na, nk, nk):
                raise ValueError("Input trispectrum shape is wrong")

            self.tsp, status = lib.set_tk3d_new_from_arrays(lk_arr, a_arr,
                                                            tkk_arr.flatten(),
                                                            int(extrap_order_lok),
                                                            int(extrap_order_lok),
                                                            int(is_logt), status)
        check(status)
        self.has_tsp = True

    def eval(self, k, a):
        status = 0

        if isinstance(k, int):
            k = float(k)
        if isinstance(k, float):
            f, status = lib.tk3d_eval_single(self.tsp, np.log(k), a, status)
        else:
            k_use = np.atleast_1d(k)
            f, status = lib.tk3d_eval_multi(self.tsp, np.log(k_use),
                                            a, k_use.size*k_use.size,
                                            status)
        check(status)
        return f

    def __del__(self):
        if hasattr(self, 'has_tsp'):
            if self.has_tsp and hasattr(self, 'tsp'):
                lib.f3d_t_free(self.tsp)
