import tempfile
import filecmp
import io

import pyccl as ccl


def test_yaml():
    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9,
                          n_s=0.97, m_nu=[0.01, 0.2, 0.3],
                          transfer_function="boltzmann_camb")

    # Make temporary files
    with tempfile.NamedTemporaryFile(delete=True) as tmpfile1, \
            tempfile.NamedTemporaryFile(delete=True) as tmpfile2:
        cosmo.write_yaml(tmpfile1.name)

        cosmo2 = ccl.Cosmology.read_yaml(tmpfile1.name)
        cosmo2.write_yaml(tmpfile2.name)

        # Compare the contents of the two files
        assert filecmp.cmp(tmpfile1.name, tmpfile2.name, shallow=False)

    cosmo = ccl.Cosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, A_s=2.1e-9,
                          n_s=0.97, m_nu=0.1, m_nu_type="equal",
                          transfer_function="boltzmann_camb")

    stream = io.StringIO()
    cosmo.write_yaml(stream)
    stream.seek(0)

    cosmo2 = ccl.Cosmology.read_yaml(stream)
    stream2 = io.StringIO()
    cosmo2.write_yaml(stream2)

    assert stream.getvalue() == stream2.getvalue()
