# fmt: off
from ase import Atoms
from ase.io import read, write


def test_magmom():

    atoms = Atoms('HH', [[.0, .0, .0], [.0, .0, .74]], pbc=True, cell=[5, 5, 5])
    atoms.set_initial_magnetic_moments([1, -1])
    moms = atoms.get_initial_magnetic_moments()
    write('test.traj', atoms)
    atoms = read('test.traj')
    assert (atoms.get_initial_magnetic_moments() == moms).all()
