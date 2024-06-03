import numpy as np
import pyccl as ccl
from pkresponse import Pmm_resp, darkemu_Pgm_resp, darkemu_Pgg_resp
import pytest

def test_Pmm_resp():
    # 设置宇宙学参数
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)
    
    # 定义输入参数
    deltah = 0.02
    lk_arr = np.log(np.geomspace(1e-4, 1e1, 100))
    a_arr = np.linspace(0.1, 1.0, 10)
    
    # 调用Pmm_resp函数
    response = Pmm_resp(cosmo, deltah=deltah, lk_arr=lk_arr, a_arr=a_arr)
    
    # 定义预期输出（可以是从之前已知结果或模拟数据）
    # 这里为了简单起见，我们假设预期输出为某个数值数组
    expected_response = np.ones_like(response) * 1e-9  # 示例预期值
    
    # 使用np.allclose验证输出
    assert np.allclose(response, expected_response, rtol=1e-5, atol=1e-8), "The response of Pmm_resp is not close to expected values."

def test_Pgm_resp():
    # 设置宇宙学参数
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)
    hmc = ccl.halos.MassFunc(ccl.halos.MassDef(200, 'critical'))
    prof_hod = ccl.halos.HaloProfileHOD()
    
    # 定义输入参数
    deltah = 0.02
    log10Mh_min = 12.0
    log10Mh_max = 15.9
    lk_arr = np.log(np.geomspace(1e-4, 1e1, 100))
    a_arr = np.linspace(0.1, 1.0, 10)
    
    # 调用darkemu_Pgm_resp函数
    response = darkemu_Pgm_resp(cosmo, hmc, prof_hod, deltah=deltah, log10Mh_min=log10Mh_min, log10Mh_max=log10Mh_max, lk_arr=lk_arr, a_arr=a_arr)
    
    # 定义预期输出（可以是从之前已知结果或模拟数据）
    # 这里为了简单起见，我们假设预期输出为某个数值数组
    expected_response = np.ones_like(response) * 1e-9  # 示例预期值
    
    # 使用np.allclose验证输出
    assert np.allclose(response, expected_response, rtol=1e-5, atol=1e-8), "The response of darkemu_Pgm_resp is not close to expected values."

def test_Pgg_resp():
    # 设置宇宙学参数
    cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)
    hmc = ccl.halos.MassFunc(ccl.halos.MassDef(200, 'critical'))
    prof_hod = ccl.halos.HaloProfileHOD()
    
    # 定义输入参数
    deltalnAs = 0.03
    log10Mh_min = 12.0
    log10Mh_max = 15.9
    lk_arr = np.log(np.geomspace(1e-4, 1e1, 100))
    a_arr = np.linspace(0.1, 1.0, 10)
    
    # 调用darkemu_Pgg_resp函数
    response = darkemu_Pgg_resp(cosmo, hmc, prof_hod, deltalnAs=deltalnAs, log10Mh_min=log10Mh_min, log10Mh_max=log10Mh_max, lk_arr=lk_arr, a_arr=a_arr)
    
    # 定义预期输出（可以是从之前已知结果或模拟数据）
    # 这里为了简单起见，我们假设预期输出为某个数值数组
    expected_response = np.ones_like(response) * 1e-9  # 示例预期值
    
    # 使用np.allclose验证输出
    assert np.allclose(response, expected_response, rtol=1e-5, atol=1e-8), "The response of darkemu_Pgg_resp is not close to expected values."

if __name__ == "__main__":
    pytest.main()