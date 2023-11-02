# pylint: skip-file
import pytest
import sys
sys.path.append("src")
from evolutionary import create_data,initialise_agents,compute_error,compute_all_errors,compute_child_coefs,create_offspring,create_generation


data_create_data = [(True,1,10),(True,3,20),(False,1,10),(False,3,20)]# future: (True,0,10),(True,1,0),(False,0,10),(False,1,0)

@pytest.mark.parametrize("bias,degree,N", data_create_data, ids=["1", "2","3","4"]) #,"5","6","7","8"
def test_create_data(bias,degree,N):
    x, y = create_data(bias,degree,N)

    assert len(x) == len(y)
    assert len(x) == N
    assert max(x) != min(x)
    assert max(y) != min(y)




#####
# pytest src/tests/utils_tests.py -v