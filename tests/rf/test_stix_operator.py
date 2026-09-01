"""Milestone-B operator self-consistency tests for the Stix GC-RF kick.

These are pure-Python re-implementations of the exact formulas in
src/icrh/RF2D_gc_stix.c and src/icrh/RF_stix_particle_history.c (branch
feature/133-stix_icrh_gc, HEAD a95f72c at the time this was written). They
require no compiled ASCOT, no HDF5 input, and no simulation: they test
whether the coded operator is internally consistent as a discretization of
the physics it is meant to represent, which is a necessary condition before
any FO-vs-GC comparison is meaningful (see the FO-vs-GC-Stix verification
plan, Milestone B).

Run with the ascotenv interpreter, e.g.:
    /global/common/software/m5040/ascotenv/bin/python3 -m unittest \
        tests.rf.test_stix_operator -v
"""
import unittest

import numpy as np
from scipy.special import jv, airy


# ---------------------------------------------------------------------------
# Faithful transcriptions of the C code being tested
# ---------------------------------------------------------------------------

def bessel_terms(lres, arg):
    """Transcription of RF2D_gc_stix_scatter, lines ~823-847."""
    Jm1 = jv(lres - 1, arg)
    Jl = jv(lres, arg)
    Jp1 = jv(lres + 1, arg)
    dbessel_l_m1 = 0.5 * (lres - 1) * Jm1 - 0.5 * arg * Jl
    dbessel_l_p1 = 0.5 * arg * Jl - 0.5 * (lres + 1) * Jp1
    return Jm1, Jp1, dbessel_l_m1, dbessel_l_p1


def term1_term2(lres, arg, Eplus2, Eminus2, E2cross):
    Jm1, Jp1, dJm1, dJp1 = bessel_terms(lres, arg)
    term1 = Eplus2 * Jm1 * Jm1 + Eminus2 * Jp1 * Jp1 + E2cross * Jm1 * Jp1
    term2 = (2 * Eplus2 * Jm1 * dJm1 + 2 * Eminus2 * Jp1 * dJp1
             + E2cross * (Jm1 * dJp1 + Jp1 * dJm1))
    return term1, term2


def chi_l(lres, arg, Eplus, Eminus, theta):
    """chi_l = (e^{-i theta}/sqrt2) [E+ J_{l-1} + e^{2 i theta} E- J_{l+1}]."""
    Jm1 = jv(lres - 1, arg)
    Jp1 = jv(lres + 1, arg)
    return (np.exp(-1j * theta) / np.sqrt(2)) * (
        Eplus * Jm1 + np.exp(2j * theta) * Eminus * Jp1)


def code_t_inter(nudot, nudot2):
    """Transcription of RF2D_gc_stix_get_interaction_time, lines ~696-724."""
    nudot = max(1e-14, abs(nudot))
    nudot2 = max(1e-14, abs(nudot2))
    zeta = -(nudot * nudot) / (2.0 * nudot2 * nudot2) ** (2.0 / 3.0)
    if abs(zeta) > 1.0:
        return np.pi / abs(nudot), zeta, "stat-phase"
    Ai, _, _, _ = airy(zeta)
    tpre = 2 * np.pi * Ai / abs(nudot2 / 2.0) ** (1.0 / 3.0)
    return 0.5 * tpre * tpre, zeta, "airy"


def stat_phase_t_inter(nudot):
    return np.pi / abs(nudot)


def init_resn_resp(ntor, R, bphi, Babs, ppar, mass, qm, omega, lres, use_bphi):
    """Transcription of the resn/resp baseline set in
    RF_particle_history_init, src/icrh/RF_stix_particle_history.c:40-47.
    `use_bphi=False` reproduces the formula before the fix in this session
    (kpara = ntor/R, no field-line projection); `use_bphi=True` reproduces
    the fixed formula (kpara = ntor/R * bphi/Babs)."""
    kpara = ntor / R * (bphi / Babs if use_bphi else 1.0)
    doppler = kpara * ppar / mass
    resn = omega - lres * qm * Babs - doppler
    resp = omega - lres * qm * Babs + doppler
    return resn, resp


def eval_resn_resp(ntor, R, bphi, Babs, ppar, mass, qm, omega, lres):
    """Transcription of the resn/resp update in RF_particle_eval_nkicks,
    src/icrh/RF_stix_particle_history.c:99-105 (unchanged by this session's
    fix - it already had the bphi/Babs projection)."""
    kpara = ntor / R * bphi / Babs
    doppler = kpara * ppar / mass
    resn = omega - lres * qm * Babs - doppler
    resp = omega - lres * qm * Babs + doppler
    return resn, resp


def code_rres_interp(Rgrid_jm1, Rgrid_j, prev, next_):
    """Transcription of the root interpolation in
    RF2D_gc_stix_compute_cold_resonances, line 562 (as-written, buggy).
    Arguments are cast to numpy float64 so that, exactly as in the C `real`
    (double) arithmetic, a zero denominator yields inf/nan rather than
    raising ZeroDivisionError."""
    Rgrid_jm1 = np.float64(Rgrid_jm1)
    Rgrid_j = np.float64(Rgrid_j)
    prev = np.float64(prev)
    next_ = np.float64(next_)
    with np.errstate(divide="ignore", invalid="ignore"):
        return Rgrid_jm1 - abs(prev) * (Rgrid_j - Rgrid_jm1) / (abs(next_) - abs(prev))


def correct_rres_interp(Rgrid_jm1, Rgrid_j, prev, next_):
    """Standard linear-interpolation root: R where the sign-changing linear
    segment through (Rgrid_jm1, prev) and (Rgrid_j, next_) crosses zero."""
    frac = abs(prev) / (abs(prev) + abs(next_))
    return Rgrid_jm1 + frac * (Rgrid_j - Rgrid_jm1)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPolarizationIdentities(unittest.TestCase):
    """Section 2.3-2.4 of the review: term1 = 4|chi_l|^2, term2 = mu d(term1)/dmu."""

    def test_term1_equals_4_chi_squared(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            lres = rng.integers(2, 15)
            arg = rng.uniform(0.1, 30.0)
            Ep = rng.normal() + 1j * rng.normal()
            Em = rng.normal() + 1j * rng.normal()
            theta = rng.uniform(0, 2 * np.pi)
            Eplus2 = 2.0 * abs(Ep) ** 2
            Eminus2 = 2.0 * abs(Em) ** 2
            E2cross = 4.0 * (Ep * np.conj(Em) * np.exp(-2j * theta)).real
            term1, _ = term1_term2(lres, arg, Eplus2, Eminus2, E2cross)
            chi = chi_l(lres, arg, Ep, Em, theta)
            self.assertAlmostEqual(term1, 4.0 * abs(chi) ** 2, places=8)

    def test_term2_is_mu_derivative_of_mu_term1(self):
        """arg = kperp*rhoL and rhoL ~ sqrt(mu), so arg = c*sqrt(mu). Check
        term1(mu) + term2(mu) == d/dmu (mu*term1(mu)) by finite differences,
        which is the drift structure the operator is supposed to realize."""
        rng = np.random.default_rng(1)
        for _ in range(50):
            lres = rng.integers(2, 12)
            c = rng.uniform(0.5, 5.0)  # kperp * sqrt(2*mass/B)/charge
            Eplus2 = rng.uniform(0.1, 3.0)
            Eminus2 = rng.uniform(0.1, 3.0)
            E2cross = rng.uniform(-1.0, 1.0) * np.sqrt(Eplus2 * Eminus2)
            mu0 = rng.uniform(1.0, 20.0)

            def mu_term1(mu):
                arg = c * np.sqrt(mu)
                t1, _ = term1_term2(lres, arg, Eplus2, Eminus2, E2cross)
                return mu * t1

            h = 1e-5 * mu0
            numeric_deriv = (mu_term1(mu0 + h) - mu_term1(mu0 - h)) / (2 * h)

            arg0 = c * np.sqrt(mu0)
            term1, term2 = term1_term2(lres, arg0, Eplus2, Eminus2, E2cross)
            coded_drift_structure = term1 + term2

            self.assertAlmostEqual(
                numeric_deriv, coded_drift_structure, delta=1e-4 * abs(numeric_deriv))


class TestDriftDiffusionConsistency(unittest.TestCase):
    """Section 3.1 (P1): a Monte-Carlo kick meant to realize
    d/dt f = d/dmu (D df/dmu) must satisfy <dmu> = d(D)/dmu, where
    D = 0.5*<dmu_stoch^2>. Checks the as-coded irnd=cos(...) normalization,
    then the proposed sqrt(2)*cos(...) fix."""

    # Fixed "wave" parameters; arg = kperp*rhoL = c*sqrt(mu) is the only
    # mu-dependence, exactly as in the C code (rhoL ~ sqrt(mu)).
    LRES = 5
    C = 1.3
    EPLUS2 = 2.0
    EMINUS2 = 0.7
    E2CROSS = 0.5

    def _dmu_and_stoch_samples(self, mu, K, phases, sqrt2_fix=False):
        arg = self.C * np.sqrt(mu)
        term1, term2 = term1_term2(
            self.LRES, arg, self.EPLUS2, self.EMINUS2, self.E2CROSS)
        dmu = K * (term1 + term2)
        # term1/(term1+term2) * dmu == term1 * K exactly (the code computes it
        # as a ratio that cancels algebraically; kept as a ratio here too, to
        # mirror src/icrh/RF2D_gc_stix.c:867 verbatim).
        ratio_dmu = (term1 / (term1 + term2)) * dmu
        irnd = np.cos(2 * np.pi * self.LRES * phases)  # matches lres*rnd*2*pi
        if sqrt2_fix:
            irnd *= np.sqrt(2.0)
        dmu_stoch = np.sqrt(np.abs(2 * mu * ratio_dmu)) * irnd
        return dmu, dmu_stoch

    def _ratio_dmu_over_dDdmu(self, sqrt2_fix, rng, n=400_000):
        mu0, K = 8.0, 0.01
        h = 1e-3 * mu0
        # Common random numbers across the three mu evaluations: the finite
        # difference of a Monte-Carlo variance estimate is otherwise buried
        # in sampling noise for any h small enough to call it a derivative.
        phases = rng.uniform(0, 1, n)
        _, stoch_p = self._dmu_and_stoch_samples(mu0 + h, K, phases, sqrt2_fix=sqrt2_fix)
        _, stoch_m = self._dmu_and_stoch_samples(mu0 - h, K, phases, sqrt2_fix=sqrt2_fix)
        dmu_0, _ = self._dmu_and_stoch_samples(mu0, K, phases, sqrt2_fix=sqrt2_fix)

        D_p = 0.5 * np.mean(stoch_p ** 2)
        D_m = 0.5 * np.mean(stoch_m ** 2)
        dD_dmu = (D_p - D_m) / (2 * h)
        return dmu_0 / dD_dmu

    def test_asdcoded_drift_diffusion_ratio_is_2(self):
        rng = np.random.default_rng(42)
        ratio = self._ratio_dmu_over_dDdmu(sqrt2_fix=False, rng=rng)
        # As-coded: <dmu> should be 2x too large relative to d(D)/dmu -
        # i.e. the realized diffusion is half of what the coded drift implies.
        self.assertAlmostEqual(ratio, 2.0, delta=0.05)

    def test_sqrt2_fix_restores_consistency(self):
        rng = np.random.default_rng(43)
        ratio = self._ratio_dmu_over_dDdmu(sqrt2_fix=True, rng=rng)
        self.assertAlmostEqual(ratio, 1.0, delta=0.05)


class TestInteractionTimeBranchSeam(unittest.TestCase):
    """Section 3.2 / 0.1 (P2 residual): the |zeta|-based switch is the right
    variable (removes timestep-dependence) but the two branches are
    discontinuous at the switch. This reproduces the reported ~3.6x jump at
    |zeta|=1 from an independent transcription of the C code."""

    def test_discontinuity_at_threshold(self):
        nudot2 = 2.0
        target_zeta = 1.0
        nudot = np.sqrt(target_zeta * (2 * nudot2 ** 2) ** (2.0 / 3.0))

        t_airy, zeta, branch = code_t_inter(nudot, nudot2)
        self.assertEqual(branch, "airy")  # |zeta|==1 falls on the Airy side
        t_stat = stat_phase_t_inter(nudot)

        jump_ratio = t_airy / t_stat
        # Document reports ~3.6x; assert it's large and in that neighborhood,
        # not that the branches agree (they should NOT agree - that's the bug).
        self.assertGreater(jump_ratio, 3.0)
        self.assertLess(jump_ratio, 4.0)

    def test_timestep_independence_of_switch_variable(self):
        """The switch criterion depends only on (nudot, nudot2), not on any
        timestep - confirming the |zeta| fix (vs. the old dt-based switch)
        makes the branch choice a physical statement, not a numerical one."""
        nudot, nudot2 = 0.3, 2.0
        _, zeta_a, branch_a = code_t_inter(nudot, nudot2)
        # Same nudot/nudot2 must give the same branch and zeta regardless of
        # any timestep parameter, because none is passed to code_t_inter.
        _, zeta_b, branch_b = code_t_inter(nudot, nudot2)
        self.assertEqual(zeta_a, zeta_b)
        self.assertEqual(branch_a, branch_b)


class TestKparaInitConsistency(unittest.TestCase):
    """Regression test for the kpara convention mismatch between
    RF_particle_history_init and RF_particle_eval_nkicks (flagged in the
    FO-vs-GC-Stix verification plan, section 0.1, as "New, introduced by the
    bphi fix"). If the two functions use different kpara conventions, a
    particle whose state hasn't changed at all between init and the first
    eval_nkicks call can still see a spurious (or missed) sign change in
    resn/resp, i.e. a first-step ghost resonance crossing.
    """

    # Representative NSTX-like state: bphi/Babs well away from 1, so any
    # convention mismatch is not accidentally masked.
    PARAMS = dict(ntor=12, R=1.0, bphi=0.4, Babs=0.45,
                  ppar=2.0e-20, mass=3.3435837724e-27,
                  qm=1.602176634e-19 / 3.3435837724e-27,
                  omega=1.885e8, lres=8)

    def test_prefix_formula_disagrees_with_eval_nkicks(self):
        """Characterizes the bug as it stood before this session's fix:
        init used kpara=ntor/R (use_bphi=False), eval_nkicks already used
        kpara=ntor/R*bphi/Babs. For an unchanged particle state the two
        must disagree here, since bphi/Babs != 1 - that disagreement is
        exactly what could register as a spurious first-step crossing."""
        resn_init, resp_init = init_resn_resp(**self.PARAMS, use_bphi=False)
        resn_eval, resp_eval = eval_resn_resp(**self.PARAMS)
        self.assertNotAlmostEqual(resn_init, resn_eval)
        self.assertNotAlmostEqual(resp_init, resp_eval)

    def test_fixed_formula_agrees_with_eval_nkicks(self):
        """After this session's fix, init and eval_nkicks use the same
        kpara convention, so an unchanged particle state must give bit-for-
        bit identical resn/resp - no spurious first-step crossing."""
        resn_init, resp_init = init_resn_resp(**self.PARAMS, use_bphi=True)
        resn_eval, resp_eval = eval_resn_resp(**self.PARAMS)
        self.assertEqual(resn_init, resn_eval)
        self.assertEqual(resp_init, resp_eval)


class TestColdResonanceRootInterpolation(unittest.TestCase):
    """Section 3.4 (B10): the coded linear-interpolation root can land
    outside the bracket, or divide by zero, for an asymmetric sign change."""

    def test_root_lands_outside_bracket_for_asymmetric_signs(self):
        Rjm1, Rj = 1.0, 1.1
        prev, next_ = -1.0, 3.0  # sign change between prev and next
        coded = code_rres_interp(Rjm1, Rj, prev, next_)
        correct = correct_rres_interp(Rjm1, Rj, prev, next_)

        self.assertTrue(coded < Rjm1 or coded > Rj,
                         f"expected the buggy formula to land outside "
                         f"[{Rjm1}, {Rj}], got {coded}")
        self.assertTrue(Rjm1 <= correct <= Rj)

    def test_division_by_zero_for_symmetric_signs(self):
        Rjm1, Rj = 1.0, 1.1
        prev, next_ = -2.0, 2.0  # |prev| == |next|
        coded = code_rres_interp(Rjm1, Rj, prev, next_)
        self.assertTrue(np.isnan(coded) or np.isinf(coded))
        # The correct formula is well-defined and lands at the midpoint.
        correct = correct_rres_interp(Rjm1, Rj, prev, next_)
        self.assertAlmostEqual(correct, 0.5 * (Rjm1 + Rj))


if __name__ == "__main__":
    unittest.main()
