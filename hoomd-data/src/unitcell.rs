//! Implement unit cell of a crystal structure.

use hoomd_geometry::shape::Hyperparallelepiped;
use hoomd_utility::valid::PositiveReal;
use hoomd_vector::Cartesian;
use itertools::Itertools;
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};

/// A unit cell of a crystal structure.
///
/// * `lattice_vectors`: The lattice vectors of the unit cell.
/// * `basis_positions`: The basis positions of the unit cell.

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UnitCell<const N: usize> {
    /// The lattice vectors of the unit cell.
    lattice_vectors: Vec<Cartesian<N>>,
    /// The basis positions of the unit cell.
    basis_positions: Vec<Cartesian<N>>,
}

impl<const N: usize> UnitCell<N> {
    /// Create a unit cell with the given lattice vectors and basis positions.
    ///
    /// * `lattice_vectors`: The lattice vectors of the unit cell.
    /// * `basis_positions`: The basis positions of the unit cell.
    pub fn new(lattice_vectors: Vec<Cartesian<N>>, basis_positions: Vec<Cartesian<N>>) -> Self {
        Self {
            lattice_vectors,
            basis_positions,
        }
    }

    /// Generate a system of particles from the unit cell.
    ///
    /// * `n_replicas`: The number of replicas of the unit cell in each dimension.
    /// * `scale`: the scale factor for the lattice vectors.
    /// * `sigma_noise`: the standard deviation of the Gaussian noise added to the particle positions.
    /// * `seed`: the seed for the random noise generation.
    ///
    /// # Example
    /// ```
    /// use hoomd_data::UnitCell;
    ///
    /// let uc = UnitCell::<3>::fcc();
    ///
    /// let (simulation_box, positions) =
    ///   uc.generate_system([4, 4, 4], 1.0.try_into()?, 0.0, 114);
    /// ```
    pub fn generate_system(
        &self,
        n_replicas: [usize; N],
        scale: PositiveReal,
        sigma_noise: f64,
        seed: u64,
    ) -> (Hyperparallelepiped<N>, Vec<Cartesian<N>>) {
        let simulation_box = Hyperparallelepiped::new(std::array::from_fn(|i| {
            self.lattice_vectors[i] * scale.get() * n_replicas[i] as f64
        }));

        let replicas = n_replicas.iter().map(|&n| 0..n).multi_cartesian_product();

        let mut rng = StdRng::seed_from_u64(seed);

        let positions = self
            .basis_positions
            .iter()
            .cartesian_product(replicas)
            .map(|(basis, replica)| {
                let mut fractional = Cartesian::from([0.0; N]);

                for i in 0..N {
                    fractional[i] = (replica[i] as f64 + basis[i]) / n_replicas[i] as f64 - 0.5;
                }

                let mut noise = Cartesian::from([0.0; N]);

                for i in 0..N {
                    let z: f64 = StandardNormal.sample(&mut rng);
                    noise[i] = z * sigma_noise;
                }

                fractional += simulation_box.fractional(noise);

                for i in 0..N {
                    fractional[i] -= (fractional[i] + 0.5).floor();
                }

                fractional
            })
            .map(|pos| simulation_box.absolute(pos))
            .collect::<Vec<_>>();

        (simulation_box, positions)
    }
}

impl UnitCell<2> {
    /// Create a graphene unit cell.
    pub fn graphene() -> Self {
        Self {
            lattice_vectors: vec![
                Cartesian::from([1.0, 0.0]),
                Cartesian::from([0.5, (3.0f64).sqrt() / 2.0]),
            ],
            basis_positions: vec![
                Cartesian::from([0.0, 0.0]),
                Cartesian::from([1.0 / 3.0, 1.0 / 3.0]),
            ],
        }
    }

    /// Create a hexagonal unit cell.
    pub fn hex() -> Self {
        Self {
            lattice_vectors: vec![
                Cartesian::from([1.0, 0.0]),
                Cartesian::from([0.5, (3.0f64).sqrt() / 2.0]),
            ],
            basis_positions: vec![Cartesian::from([0.0, 0.0])],
        }
    }

    /// Create an oblique unit cell with the given aspect ratio and angle.
    pub fn oblique(aspect_ratio: PositiveReal, angle: PositiveReal) -> Self {
        Self {
            lattice_vectors: vec![
                Cartesian::from([1.0, 0.0]),
                Cartesian::from([
                    aspect_ratio.get() * angle.get().cos(),
                    aspect_ratio.get() * angle.get().sin(),
                ]),
            ],
            basis_positions: vec![Cartesian::from([0.0, 0.0])],
        }
    }

    /// Create a square unit cell.
    pub fn square() -> Self {
        Self {
            lattice_vectors: vec![Cartesian::from([1.0, 0.0]), Cartesian::from([0.0, 1.0])],
            basis_positions: vec![Cartesian::from([0.0, 0.0])],
        }
    }
}

impl UnitCell<3> {
    /// Create a body-centered cubic (BCC) unit cell.
    pub fn bcc() -> Self {
        Self {
            lattice_vectors: vec![
                Cartesian::from([1.0, 0.0, 0.0]),
                Cartesian::from([0.0, 1.0, 0.0]),
                Cartesian::from([0.0, 0.0, 1.0]),
            ],
            basis_positions: vec![
                Cartesian::from([0.0, 0.0, 0.0]),
                Cartesian::from([0.5, 0.5, 0.5]),
            ],
        }
    }

    /// Create a face-centered cubic (FCC) unit cell.
    pub fn fcc() -> Self {
        Self {
            lattice_vectors: vec![
                Cartesian::from([1.0, 0.0, 0.0]),
                Cartesian::from([0.0, 1.0, 0.0]),
                Cartesian::from([0.0, 0.0, 1.0]),
            ],
            basis_positions: vec![
                Cartesian::from([0.0, 0.0, 0.0]),
                Cartesian::from([0.5, 0.5, 0.0]),
                Cartesian::from([0.5, 0.0, 0.5]),
                Cartesian::from([0.0, 0.5, 0.5]),
            ],
        }
    }

    /// Create a hexagonal close-packed (HCP) unit cell.
    pub fn hcp(c: PositiveReal) -> Self {
        Self {
            lattice_vectors: vec![
                Cartesian::from([1.0, 0.0, 0.0]),
                Cartesian::from([0.5, (3.0f64).sqrt() / 2.0, 0.0]),
                Cartesian::from([0.0, 0.0, c.get()]),
            ],
            basis_positions: vec![
                Cartesian::from([0.0, 0.0, 0.0]),
                Cartesian::from([1.0 / 3.0, 1.0 / 3.0, 1.0 / 2.0]),
            ],
        }
    }

    /// Create a simple cubic (SC) unit cell.
    pub fn sc() -> Self {
        Self {
            lattice_vectors: vec![
                Cartesian::from([1.0, 0.0, 0.0]),
                Cartesian::from([0.0, 1.0, 0.0]),
                Cartesian::from([0.0, 0.0, 1.0]),
            ],
            basis_positions: vec![Cartesian::from([0.0, 0.0, 0.0])],
        }
    }
}
