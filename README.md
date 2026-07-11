# TWIST-J

[![ORCID](https://img.shields.io/badge/ORCID-0009--0008--5463--278X-brightgreen)](https://orcid.org/0009-0008-5463-278X)

**A. M. Thorn** · [twistj.com](https://twistj.com) · [Manifest](https://twistj.com/manifest/) · [Canon](https://twistj.com/canon/)

> A finite rule, unbounded becoming.

The whole universe, matter, both forces, space and time, from a single step:

```
j = zeta_5,   the primitive fifth root of unity,   j^5 = 1
J = 1 + j^2
```

Computed entirely in integers. Zero free dimensionless parameters. One SI calibration anchor, the electron mass, and that only to fix units.

> If it cannot be calculated in integers, it is not physics.

You can find quick canon overview at [hub](https://twistj.com/canon/core/)
---

## What TWIST-J is

- **One axiom.** J = 1 + zeta_5^2 in the cyclotomic field Q(zeta_5). Everything derives from this single algebraic unit. J is the verb; phi and pi are projections of J, not inputs.
- **Two projections, two forces.** The modulus, |J| = 1/phi, gives gravity and scale. The argument, arg(J) = 2 pi / 5, gives electromagnetism and phase.
- **One engine.** Multiplication by J is a 4x4 integer matrix, entries only {-1, 0, 1}, determinant 1, trace 3. Four additions per step, zero multiplications. No drift; exact integer arithmetic, verified bit by bit on two architectures with a SHA-256 over every result.
- **The plenum, not the vacuum.** What is called empty space is a full, counting substance. Time is a counter. Space is a commutator. Curvature is what remains. The total energy sums to zero, as a theorem of the closed whole, not as a wish.

---

## The Messenger (main script)

**Script:** `scripts/j_twist_messenger.py`

The Messenger is the core public artifact of this repository. It is not a simulation and not a numerical fit. It is a deterministic derivation chain from the single axiom, with every identity asserted at runtime.

Run it:

```
python scripts/j_twist_messenger.py
```

It prints the axiom, the derivation chain, the asserted identities, a comparison table against experiment, and the falsifiability criteria. Two voices meeting on one axiom, joined by an exact rational bridge:

- **Analytical (continuous).** phi, the gyron, the quantum angle, the mass ladder, gravity, cosmology.
- **Binary kernel (discrete).** Dynamics on Z_5^6. State updates use addition and subtraction only, entries {-1, 0, 1}, driven by the Thue-Morse clock. Verified over all 15625 states in exact integer arithmetic.
- **Zeta layer (the bridge).** Bernoulli numbers, L-functions, Dedekind zeta. It shows the icosahedral invariants are the spectral data of Q(zeta_5).

The continuous and the discrete agree. No fitting. No parameters. No random. It speaks, it checks, it ends.

---

## What it derives, and how it stands

The chain is dimensionless. The electron mass m_e is used only as a unit anchor to reach SI; the Z-boson mass M_Z appears only in the optional Extended block. Reference values are CODATA 2022 and PDG 2024. Both relative error and sigma are printed, tension included, never hidden.

- **Fine structure constant.** Derived purely from J. It agrees with the measured value to about a tenth of a part per billion. It was never aimed there; it fell out of the computation. This is the strongest piece.
- **Proton to electron mass ratio.** 6 pi^5 with a small correction, off by about one part per million. That remainder is admitted, not celebrated; it is expected to be QCD binding, which is not yet derived from J.
- **Dark energy.** A fixed prediction, the equation of state w = -14/15, now standing trial before the DESI data.
- **Leptons, electroweak mixing, a dimensionless gravitational coupling.** Stated as explicit ratios. Genuine successes and structural approximations are each marked as what they are.

---

## Structure

- Step group: the binary icosahedral group 2I = SL(2,5). Through the McKay correspondence it leads to the affine Dynkin diagram tilde-E8.
- Stable classes: exactly 313 attractors, 125 + 125 + 62 + 1.
- State space: Z_5^6, that is 15625 = 125^2 checkpoints.

Carried with explicit status as a model-internal classification, not asserted as a general theorem.

---

## Uniqueness (model-internal)

Under the model's extension rules, across the Platonic {p, q} solids, only the icosahedral case {3, 5} reproduces sub-ppb agreement for alpha^(-1) together with ppm-level agreement for m_p/m_e. This is a model-internal uniqueness result, not a general mathematical theorem.

---

## Falsifiability

Every claim carries an explicit status: theorem, definition, axiom, identity, computation, open wall, falsified. Falsification is first-class progress, and no summary is stronger than its own label.

Kill shots, armed:

- electroweak mixing sin^2(theta_W) outside its stated window,
- a nonzero photon mass,
- a fourth sequential fermion generation,
- O-TT-GAUGE-DERIVATION failing its falsifier. The exponential metric was already refuted by the ringdown of gravitational wave GW250114; a closed gravitational wave sector was derived the same day, and this one derivation remains owed.

Live trial:

- dark energy w = -14/15, now facing DESI.

If any kill shot fires, the model is falsified.

---

## A by-product: JAM

JAM is a small language model built to test whether the J-algebra helps as a computational layer. Almost everything failed: a fixed linear transform placed before a learned layer is absorbed into it without remainder. Two things survived, the DJ motor and the BinDJ positional encoding. Inference runs in INT4. The negative result is recorded with the same weight as the positive ones.

---

## Selected work

**The Plenum: A New Vacuum.** A conceptual and mathematical exploration of a discrete mechanical substrate underlying spacetime. Zenodo DOI: https://doi.org/10.5281/zenodo.18356446

Full ledger and papers: [the Canon](https://twistj.com/canon/) and [ORCID](https://orcid.org/0009-0008-5463-278X).

---

## Links

- Hub: https://twistj.com
- Manifest: https://twistj.com/manifest/
- Canon: https://twistj.com/canon/
- ORCID: https://orcid.org/0009-0008-5463-278X
- X: [@amthorn73](https://x.com/amthorn73)
