"""AFSI5: Versatile fusion source integrator AFSI for fast ion and neutron
studies in fusion devices
"""
import ctypes
from typing import List
import numpy as np
import unyt
import numpy.ctypeslib as npctypes
from pathlib import Path

from a5py.ascot5io.dist import DistData
from a5py.ascotpy.libascot import _LIBASCOT, STRUCT_HIST, STRUCT_AFSIDATA, \
    PTR_REAL, AFSI_REACTIONS
from a5py.ascotpy.ascot2py import hist_coordinate__enumvalues
from a5py.exceptions import AscotNoDataException
from a5py.physlib.units import parseunits
from a5py.routines.distmixin import DistMixin

try:
    import desc
except ImportError:
    desc = None

class Afsi():
    """ASCOT Fusion Source Integrator AFSI.

    ASCOT Fusion Source Integrator (AFSI) is a tool for calculating fusion rates
    and fusion products for arbitrary particle populations. It can be used
    either as a preprocessing tool e.g. to generate source distribution
    of alphas or as a postprocessing tool e.g. to evaluate neutron source
    from beam-thermal fusion.

    AFSI can be used in three ways:

    - thermal: Calculates fusion rates between two Maxwellian populations.
    - beam-thermal: Calculates fusion rates between a Maxwellian and arbitrary
      population e.g. beam ions as obtained from ASCOT5 simulation.
    - beam-beam: Calculates fusion rates between two arbitrary populations.

    Reference:
    https://iopscience.iop.org/article/10.1088/1741-4326/aa92e9

    Attributes
    ----------
    _ascot : :class:`.Ascot`
        Ascot object used to run AFSI.
    """

    def __init__(self, ascot):
        self._ascot = ascot

    @parseunits(rho='dimensionless', phimin='deg', phimax='deg',
                ekin1='eV', ekin2='eV', pitch1='dimensionless',
                pitch2='dimensionless', strip=False)
    def thermal_from_desc(self, reaction, descfn, rho=None, 
                          phimin=0.0, phimax=360.0*unyt.deg,
                          nphi: int = 2,
                          ekin1=None, pitch1=None,
                          ekin2=None, pitch2=None,
                          nmc=1000
                          ) -> List[DistData]:
        """
        Calculate thermonuclear fusion between two thermal (Maxwellian)
        species.

        This variant is used to initialize the AFSI run using the DESC
        calculation of the volume, which is more accurate and stable than
        the one implemented in ASCOT.

        @todo: the theta input in DESC is actually different from the one
        in ASCOT, and some transformations are here required. Right now
        it just assumes theta = (0, 360) deg.
        
        Parameters
        ----------
        reaction : int or str
            Fusion reaction index or name.
        rho : array_like
            Abscissa for the radial coordinate in (rho,theta,phi) basis.
        phimin : float
            Minimum toroidal angle in degrees.
        phimax : float
            Maximum toroidal angle in degrees.
        ekin1 : array_like
            Abscissa for the kinetic energy in (E,pitch) basis for the first
            reaction product.
        pitch1 : array_like
            Abscissa for the pitch in (E,pitch) basis for the first reaction
            product.
        ekin2 : array_like
            Abscissa for the kinetic energy in (E,pitch) basis for the second
            reaction product.
        pitch2 : array_like
            Abscissa for the pitch in (E,pitch) basis for the second reaction
            product.
        nmc : int, optional
            Number of MC samples used in each (R, phi, z) bin.
        Returns
        -------
        prod1 : DistData
            Source distribution of the first fusion product.
        prod2 : DistData
            Source distribution of the second fusion product.
        """
        # We generate the limits in theta:
        phi = np.linspace(phimin.to('deg').value, 
                          phimax.to('deg').value, nphi) * unyt.deg
        theta = np.linspace(0, 360, 2) * unyt.deg
        
        # Getting the masses of the inputs reactants and checking whether
        # they are present in the plasma input.
        m1, q1, m2, q2, _, qprod1, _, qprod2, _ = self.reactions(reaction)
        reactions = {v: k for k, v in AFSI_REACTIONS.items()}
        reaction = reactions[reaction]
        anum1 = np.round(m1.to("amu").v)
        anum2 = np.round(m2.to("amu").v)
        znum1 = np.round(q1.to("e").v)
        znum2 = np.round(q2.to("e").v)
        q1 = np.round(qprod1.to("e").v)
        q2 = np.round(qprod2.to("e").v)

        nspec, _, _, anums, znums = self._ascot.input_getplasmaspecies()
        ispecies1, ispecies2 = np.nan, np.nan
        for i in np.arange(nspec):
            if( anum1 == anums[i] and znum1 == znums[i] ):
                ispecies1 = i
            if( anum2 == anums[i] and znum2 == znums[i] ):
                ispecies2 = i
        if np.isnan(ispecies1) or np.isnan(ispecies2):
            self._ascot.input_free(bfield=True, plasma=True)
            raise ValueError("Reactant species not present in plasma input.")
        mult = 0.5 if ispecies1 == ispecies2 else 1.0

        # We initialize the histograms.
        prod1 = self._init_histogram(
                    rho.v, theta.to('rad').v, 
                    phi.to('rad').v, 
                    ekin1.to('eV').v, 
                    pitch1.to('dimensionless').v,
                    charge=q1, exi=True, toroidal=True)
        prod2 = self._init_histogram(
                    rho.v, theta.to('rad').v, 
                    phi.to('rad').v, 
                    ekin2.to('eV').v, 
                    pitch2.to('dimensionless').v,
                    charge=q2, exi=True, toroidal=True)
        
        # We need now to compute the volume and the center of the cell using the
        # the DESC file.
        if isinstance(descfn, str) or isinstance(descfn, Path):
            if desc is None:
                raise ImportError("The 'desc' module is required to use "
                                  "the 'thermal_from_desc' method.")
            fam = desc.io.load(descfn)
            try:  # if file is an EquilibriaFamily, use final Equilibrium
                eq = fam[-1]
            except:  # file is already an Equilibrium
                eq = fam
        else:
            eq = descfn

        if not hasattr(eq, 'compute'):
            raise ValueError("The 'descfn' argument must be either a file "
                             "name or a DESC Equilibrium object.")

        # Computing the volume using DESC.
        grid = desc.grid.LinearGrid(rho=rho.v, M=eq.M_grid, N=eq.N_grid, 
                                    NFP=eq.NFP, sym=False)
        data = eq.compute("V(r)", grid=grid)
        vol = np.diff(np.array(grid.compress(data["V(r)"]))) # Volume contained within each rho shell

        # We need to scale the volume to the actual phi range used.
        # The following is only true if
        # - Axisymmetry (tokamaks).
        # - The input grid has that allows to reduce the phi range to [phimin, phimax].
        # @TODO: Generalize this properly.
        dphi = (phimax - phimin).to('rad').value / (2 * np.pi)
        vol = vol * dphi

        # The volume should have the same shape of the (rho.size, theta.size, phi.size)
        vol = vol[:, np.newaxis, np.newaxis] * np.ones((1, theta.size-1, phi.size-1))

        # We now compute the center of the coordinates in cylindrical.
        rhoc = 0.5 * (rho[:-1] + rho[1:])
        thetac = 0.5 * (theta[:-1] + theta[1:])
        phic = 0.5 * (phi[:-1] + phi[1:]) # This is already in cylindrical.

        rr, tt, pp = np.meshgrid(rhoc.v, thetac.to('rad').v, phic.to('rad').v,
                                 indexing='ij')

        grid = desc.grid.Grid(np.stack([rr, tt, pp], axis=-1))
        data = eq.compute(["R", "phi", "Z"], grid=grid)
        rc = np.array(grid.compress(data["R"]))
        zc = np.array(grid.compress(data["Z"]))
        phic = np.array(grid.compress(data["phi"]))

        for ii in [rc, zc, phic, vol]:
            if not ii.flags["C_CONTIGUOUS"]:
                ii = np.ascontiguousarray(ii)
        
        # We initialize the AFSI data structure.
        afsi = self._init_afsi_data(
            react1=ispecies1, react2=ispecies2, reaction=reaction, mult=mult,
            r=rc, phi=phic, z=zc, vol=vol,
            )

        if self._ascot._mute == 'err':
            verbose = 2
        elif self._ascot._mute == 'yes':
            verbose = 3
        else:
            verbose = 1
        
        _LIBASCOT.afsi_run(ctypes.byref(self._ascot._sim),
                           ctypes.byref(afsi), nmc, prod1, prod2,
                           verbose, 0)
        self._ascot.input_free(bfield=True, plasma=True)

        # Reload Ascot
        # self._ascot.file_load(self._ascot.file_getpath())
        prod1 = self._build_distdata(prod1)
        prod2 = self._build_distdata(prod2)
        return prod1, prod2

    def thermal(
            self,
            reaction,
            r=None,
            phi=None,
            z=None,
            rho=None,
            theta=None,
            ppar1=None,
            pperp1=None,
            ekin1=None,
            pitch1=None,
            ppar2=None,
            pperp2=None,
            ekin2=None,
            pitch2=None,
            nmc=1000,
            ):
        """Calculate thermonuclear fusion between two thermal (Maxwellian)
        species.

        Parameters
        ----------
        reaction : int or str
            Fusion reaction index or name.
        r : array_like
            Abscissa for the radial coordinate in (R,phi,z) basis.
        phi : array_like
            Abscissa for the toroidal coordinate in (R,phi,z) and
            (rho,theta,phi) basis.
        z : array_like
            Abscissa for the axial coordinate in (R,phi,z) basis.
        rho : array_like
            Abscissa for the radial coordinate in (rho,theta,phi) basis.
        theta : array_like
            Abscissa for the poloidal coordinate in (rho,theta,phi) basis.
        ppar1 : array_like
            Abscissa for the parallel momentum in (ppar,pperp) basis for the
            first reaction product.
        pperp1 : array_like
            Abscissa for the perpendicular momentum in (ppar,pperp) basis for
            the first reaction product.
        ekin1 : array_like
            Abscissa for the kinetic energy in (E,pitch) basis for the first
            reaction product.
        pitch1 : array_like
            Abscissa for the pitch in (E,pitch) basis for the first reaction
            product.
        ppar2 : array_like
            Abscissa for the parallel momentum in (ppar,pperp) basis for the
            second reaction product.
        pperp2 : array_like
            Abscissa for the perpendicular momentum in (ppar,pperp) basis for
            the second reaction product.
        ekin2 : array_like
            Abscissa for the kinetic energy in (E,pitch) basis for the second
            reaction product.
        pitch2 : array_like
            Abscissa for the pitch in (E,pitch) basis for the second reaction
            product.
        nmc : int, optional
            Number of MC samples used in each (R, phi, z) bin.

        Returns
        -------
        prod1 : array_like
            Source distribution of the first fusion product.
        prod2 : array_like
            Source distribution of the second fusion product.
        """
        if phi is None:
            phi = np.array([0, 360])
        dont_use_rpz_basis = ( any([r is None, z is None]) and
                               all([r is None, z is None]) )
        use_rpz_basis = ( not any([r is None, z is None]) and
                          all([not r is None, not z is None]) )
        if use_rpz_basis and dont_use_rpz_basis:
            raise ValueError(
                "Either give all of r, phi, and z to use (R,phi,z) basis or "
                "use (rho,theta,phi) basis instead"
                )
        dont_use_ekinpitch_basis = ( any([ekin1 is None, pitch1 is None]) and
                                     all([ekin1 is None, pitch1 is None]) )
        use_ekinpitch_basis = ( not any([ekin1 is None, pitch1 is None]) and
                                all([not ekin1 is None, not pitch1 is None]) )
        if use_ekinpitch_basis and dont_use_ekinpitch_basis:
            raise ValueError(
                "Either give both ekin1 and pitch1 to use (E,pitch) basis or "
                "use (ppar,pperp) basis instead"
                )

        dont_use_ekinpitch_basis2 = ( any([ekin2 is None, pitch2 is None]) and
                                      all([ekin2 is None, pitch2 is None]) )
        use_ekinpitch_basis2 = ( not any([ekin2 is None, pitch2 is None]) and
                                 all([not ekin2 is None, not pitch2 is None]) )
        if use_ekinpitch_basis2 and dont_use_ekinpitch_basis2:
            raise ValueError(
                "Either give both ekin2 and pitch2 to use (E,pitch) basis or "
                "use (ppar,pperp) basis instead"
                )
        if use_ekinpitch_basis != use_ekinpitch_basis2:
            if use_ekinpitch_basis:
                raise ValueError(
                    "Both product distributions have to use same basis system: "
                    "now the first product were to use (E,pitch) and the "
                    "second (ppar,pperp)"
                    )
            else:
                raise ValueError(
                    "Both product distributions have to use same basis system: "
                    "now the second product were to use (E,pitch) and the "
                    "first (ppar,pperp)"
                    )
        self._ascot.input_init(bfield=True, plasma=True)
        if dont_use_rpz_basis:
            rho = np.linspace(0, 1, 10) if rho is None else rho
            theta = np.array([0, 180, 360]) if theta is None else theta
            vol, rc, phic, zc = self._ascot.input_rhovolume(
                nrho=rho.size, ntheta=theta.size, nphi=phi.size, method="prism",
                return_coords=True, minrho=rho[0], maxrho=rho[-1], minphi=phi[0], maxphi=phi[-1], mintheta=theta[0], maxtheta=theta[-1]
                )
            phic = phic.ravel()
        else:
            phic, rc, zc = np.meshgrid(0.5*(phi[:-1]+phi[1:]),
                                        0.5*(r[:-1]+r[1:]),
                                        0.5*(z[:-1]+z[1:]))
            phic *= np.pi/180
            vol = ( rc * np.diff(r[:2]) * np.diff(z[:2]) * np.diff(phi[:2])
                       * np.pi/180 )
        if dont_use_ekinpitch_basis:
            ppar1 = 1.3e-19 * np.linspace(-1., 1., 80) if ppar1 is None else ppar1
            pperp1 = np.linspace(0, 1.3e-19, 40) if pperp1 is None else pperp1
            ppar2 = 1.3e-19 * np.linspace(-1., 1., 80) if ppar2 is None else ppar2
            pperp2 = np.linspace(0, 1.3e-19, 40) if pperp2 is None else pperp2

        m1, q1, m2, q2, _, qprod1, _, qprod2, _ = self.reactions(reaction)
        reactions = {v: k for k, v in AFSI_REACTIONS.items()}
        reaction = reactions[reaction]
        anum1 = np.round(m1.to("amu").v)
        anum2 = np.round(m2.to("amu").v)
        znum1 = np.round(q1.to("e").v)
        znum2 = np.round(q2.to("e").v)
        q1 = np.round(qprod1.to("e").v)
        q2 = np.round(qprod2.to("e").v)

        nspec, _, _, anums, znums = self._ascot.input_getplasmaspecies()
        ispecies1, ispecies2 = np.nan, np.nan
        for i in np.arange(nspec):
            if( anum1 == anums[i] and znum1 == znums[i] ):
                ispecies1 = i
            if( anum2 == anums[i] and znum2 == znums[i] ):
                ispecies2 = i
        if np.isnan(ispecies1) or np.isnan(ispecies2):
            self._ascot.input_free(bfield=True, plasma=True)
            raise ValueError("Reactant species not present in plasma input.")
        mult = 0.5 if ispecies1 == ispecies2 else 1.0

        afsi = self._init_afsi_data(
            react1=ispecies1, react2=ispecies2, reaction=reaction, mult=mult,
            r=rc, phi=phic, z=zc, vol=vol,
            )

        if use_ekinpitch_basis:
            if use_rpz_basis:
                prod1 = self._init_histogram(
                    r, phi*np.pi/180, z, ekin1, pitch1, charge=q1, exi=True)
                prod2 = self._init_histogram(
                    r, phi*np.pi/180, z, ekin2, pitch2, charge=q2, exi=True)
            else:
                prod1 = self._init_histogram(
                    rho, theta*np.pi/180, phi*np.pi/180, ekin1, pitch1,
                    charge=q1, exi=True, toroidal=True)
                prod2 = self._init_histogram(
                    rho, theta*np.pi/180, phi*np.pi/180, ekin2, pitch2,
                    charge=q2, exi=True, toroidal=True)
        else:
            if use_rpz_basis:
                prod1 = self._init_histogram(
                    r, phi*np.pi/180, z, ppar1, pperp1, charge=q1, exi=False)
                prod2 = self._init_histogram(
                    r, phi*np.pi/180, z, ppar2, pperp2, charge=q2, exi=False)
            else:
                prod1 = self._init_histogram(
                    rho, theta*np.pi/180, phi*np.pi/180, ppar1, pperp1,
                    charge=q1, exi=False, toroidal=True)
                prod2 = self._init_histogram(
                    rho, theta*np.pi/180, phi*np.pi/180, ppar2, pperp2,
                    charge=q2, exi=False, toroidal=True)

        if self._ascot._mute == 'err':
            verbose = 2
        elif self._ascot._mute == 'yes':
            verbose = 3
        else:
            verbose = 1
        
        _LIBASCOT.afsi_run(ctypes.byref(self._ascot._sim),
                           ctypes.byref(afsi), nmc, prod1, prod2,
                           verbose, 0)

        self._ascot.input_free(bfield=True, plasma=True)

        # Reload Ascot - not needed anymore.
        # self._ascot.file_load(self._ascot.file_getpath())
        prod1 = self._build_distdata(prod1)
        prod2 = self._build_distdata(prod2)
        return prod1, prod2

    def beamthermal(
            self,
            reaction,
            beam,
            swap=False,
            nmc=1000,
            ppar1=None,
            pperp1=None,
            ppar2=None,
            pperp2=None,
            ):
        """Calculate beam-thermal fusion.

        Parameters
        ----------
        reaction : int
            Fusion reaction index
        beam : dict
            Beam distribution that acts as the first reactant.
        swap : bool, optional
            If True, beam distribution acts as the second reactant and
            the first reactant is a background species.
        nmc : int, optional
            Number of MC samples used in each (R, phi, z) bin.
        ppar1 : array_like
            Abscissa for the parallel momentum in (ppar,pperp) basis for the
            first reaction product.
        pperp1 : array_like
            Abscissa for the perpendicular momentum in (ppar,pperp) basis for
            the first reaction product.
        ppar2 : array_like
            Abscissa for the parallel momentum in (ppar,pperp) basis for the
            second reaction product.
        pperp2 : array_like
            Abscissa for the perpendicular momentum in (ppar,pperp) basis for
            the second reaction product.

        Returns
        -------
        prod1 : array_like
            Fusion product 1 distribution.
        prod2 : array_like
            Fusion product 2 distribution.
        """
        ppar1 = 1.3e-19 * np.linspace(-1., 1., 50) if ppar1 is None else ppar1
        pperp1 = np.linspace(0, 1.3e-19, 50) if pperp1 is None else pperp1
        ppar2 = 1.3e-19 * np.linspace(-1., 1., 50) if ppar2 is None else ppar2
        pperp2 = np.linspace(0, 1.3e-19, 50) if pperp2 is None else pperp2

        m1, q1, m2, q2, _, qprod1, _, qprod2, _ = self.reactions(reaction)
        reactions = {v: k for k, v in AFSI_REACTIONS.items()}
        reaction = reactions[reaction]
        anum1 = np.round(m1.to("amu").v)
        anum2 = np.round(m2.to("amu").v)
        znum1 = np.round(q1.to("e").v)
        znum2 = np.round(q2.to("e").v)
        q1 = np.round(qprod1.to("e").v)
        q2 = np.round(qprod2.to("e").v)

        self._ascot.input_init(bfield=True, plasma=True)
        nspec, _, _, anums, znums = self._ascot.input_getplasmaspecies()
        ispecies = np.nan
        for i in np.arange(nspec):
            if( swap and anum1 == anums[i] and znum1 == znums[i] ):
                ispecies = i
                react1 = ispecies
                react2 = self._init_dist_5d(beam)
            if( not swap and anum2 == anums[i] and znum2 == znums[i] ):
                ispecies = i
                react2 = ispecies
                react1 = self._init_dist_5d(beam)
        if np.isnan(ispecies):
            self._ascot.input_free(bfield=True, plasma=True)
            raise ValueError("Reactant species not present in plasma input.")

        mult = 1.0
        r, z, phi = ( beam.abscissa_edges("r"), beam.abscissa_edges("z"),
                      beam.abscissa_edges("phi").to("rad") )
        phic, rc, zc = np.meshgrid(0.5*(phi[:-1]+phi[1:]),
                                        0.5*(r[:-1]+r[1:]),
                                        0.5*(z[:-1]+z[1:]))
        vol = ( rc * np.diff(r[:2]) * np.diff(z[:2]) * np.diff(phi[:2]) )
        afsi = self._init_afsi_data(
            react1=react1, react2=react2, reaction=reaction, mult=mult,
            r=rc, phi=phic, z=zc, vol=vol,
            )

        prod1 = self._init_histogram(
            beam.abscissa_edges("r"),
            beam.abscissa_edges("phi").to("rad"),
            beam.abscissa_edges("z"),
            ppar1,
            pperp1,
            charge=q1,
            exi=False,
            )
        prod2 = self._init_histogram(
            beam.abscissa_edges("r"),
            beam.abscissa_edges("phi").to("rad"),
            beam.abscissa_edges("z"),
            ppar2,
            pperp2,
            charge=q2,
            exi=False,
            )

        _LIBASCOT.afsi_run(ctypes.byref(self._ascot._sim),
                            ctypes.byref(afsi), nmc, prod1, prod2,
                            )
        self._ascot.input_free(bfield=True, plasma=True)

        # Reload Ascot
        self._ascot.file_load(self._ascot.file_getpath())
        return prod1, prod2

    def beambeam(
            self,
            reaction,
            beam1,
            beam2=None,
            nmc=1000,
            ppar1=None,
            pperp1=None,
            ppar2=None,
            pperp2=None,
            ):
        """Calculate beam-beam fusion.

        Parameters
        ----------
        reaction : int
            Fusion reaction index.
        beam1 : dict
            Beam1 distribution.
        beam2 : dict, optional
            Beam2 distribution or None to calculate fusion generation with
            beam1 itself.
        nmc : int, optional
            Number of MC samples used in each (R, phi, z) bin.
        ppar1 : array_like
            Abscissa for the parallel momentum in (ppar,pperp) basis for the
            first reaction product.
        pperp1 : array_like
            Abscissa for the perpendicular momentum in (ppar,pperp) basis for
            the first reaction product.
        ppar2 : array_like
            Abscissa for the parallel momentum in (ppar,pperp) basis for the
            second reaction product.
        pperp2 : array_like
            Abscissa for the perpendicular momentum in (ppar,pperp) basis for
            the second reaction product.

        Returns
        -------
        prod1 : array_like
            Fusion product 1 distribution.
        prod2 : array_like
            Fusion product 2 distribution.
        """
        ppar1 = 1.3e-19 * np.linspace(-1., 1., 80) if ppar1 is None else ppar1
        pperp1 = np.linspace(0, 1.3e-19, 40) if pperp1 is None else pperp1
        ppar2 = 1.3e-19 * np.linspace(-1., 1., 80) if ppar2 is None else ppar2
        pperp2 = np.linspace(0, 1.3e-19, 40) if pperp2 is None else pperp2
        _, _, _, _, _, qprod1, _, qprod2, _ = self.reactions(reaction)
        reactions = {v: k for k, v in AFSI_REACTIONS.items()}
        reaction = reactions[reaction]
        q1 = np.round(qprod1.to("e").v)
        q2 = np.round(qprod2.to("e").v)

        self._ascot.input_init(bfield=True, plasma=True)

        r, z, phi = ( beam1.abscissa_edges("r"), beam1.abscissa_edges("z"),
                      beam1.abscissa_edges("phi").to("rad") )
        phic, rc, zc = np.meshgrid(0.5*(phi[:-1]+phi[1:]),
                                        0.5*(r[:-1]+r[1:]),
                                        0.5*(z[:-1]+z[1:]))
        vol = ( rc * np.diff(r[:2]) * np.diff(z[:2]) * np.diff(phi[:2]) )

        react1 = self._init_dist_5d(beam1)
        if beam2 is not None:
            react2 = self._init_dist_5d(beam2)
            mult = 1.0
        else:
            react2 = react1
            mult = 0.5
        afsi = self._init_afsi_data(
            react1=react1, react2=react2, reaction=reaction, mult=mult,
            r=rc, phi=phic, z=zc, vol=vol,
            )

        prod1 = self._init_histogram(
            beam1.abscissa_edges("r"),
            beam1.abscissa_edges("phi").to("rad"),
            beam1.abscissa_edges("z"),
            ppar1,
            pperp1,
            charge=q1,
            exi=False,
            )
        prod2 = self._init_histogram(
            beam1.abscissa_edges("r"),
            beam1.abscissa_edges("phi").to("rad"),
            beam1.abscissa_edges("z"),
            ppar2,
            pperp2,
            charge=q2,
            exi=False,
            )
        _LIBASCOT.afsi_run(ctypes.byref(self._ascot._sim),
                           ctypes.byref(afsi), nmc, prod1, prod2,
                           )

        self._ascot.input_free(bfield=True, plasma=True)

        # Reload Ascot
        self._ascot.file_load(self._ascot.file_getpath())
        return prod1, prod2

    def _init_afsi_data(self, react1, react2, reaction, mult, r, phi, z, vol):
        afsidata = STRUCT_AFSIDATA()
        afsidata.reaction = reaction
        afsidata.mult = mult
        afsidata.r = r.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        afsidata.z = z.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        afsidata.phi = phi.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        afsidata.vol = vol.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        afsidata.volshape[:] = vol.shape
        if isinstance(react1, np.int_):
            afsidata.type1 = 2
            afsidata.thermal1 = react1
        else:
            afsidata.type1 = 1
            afsidata.beam1 = ctypes.pointer(react1)
        if isinstance(react2, np.int_):
            afsidata.type2 = 2
            afsidata.thermal2 = react2
        else:
            afsidata.type2 = 1
            afsidata.beam2 = ctypes.pointer(react2)
        return afsidata

    def _init_dist_5d(self, dist):
        data = STRUCT_HIST()
        coordinates = np.array([0, 1, 2, 5, 6, 14, 15], dtype="uint32")
        nbin = np.array([
            dist.abscissa("r").size, dist.abscissa("phi").size,
            dist.abscissa("z").size, dist.abscissa("ppar").size,
            dist.abscissa("pperp").size, dist.abscissa("time").size,
            dist.abscissa("charge").size], dtype="u8")
        binmin = np.array([
            dist.abscissa_edges("r")[0], dist.abscissa_edges("phi").to("rad")[0],
            dist.abscissa_edges("z")[0], dist.abscissa_edges("ppar")[0],
            dist.abscissa_edges("pperp")[0], dist.abscissa_edges("time")[0],
            dist.abscissa_edges("charge")[0]])
        binmax = np.array([
            dist.abscissa_edges("r")[-1], dist.abscissa_edges("phi").to("rad")[-1],
            dist.abscissa_edges("z")[-1], dist.abscissa_edges("ppar")[-1],
            dist.abscissa_edges("pperp")[-1], dist.abscissa_edges("time")[-1],
            dist.abscissa_edges("charge")[-1]])
        _LIBASCOT.hist_init(
            ctypes.byref(data),
            coordinates.size,
            coordinates.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
            binmin.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            binmax.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            nbin.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t))
            )
        d = dist.histogram().ravel()
        for i in range(data.nbin):
            data.bins[i] = d[i]
        return data

    def _init_histogram(self, *args, charge=None, time=None, exi=False,
                        toroidal=False):
        prod = STRUCT_HIST()
        if time is None:
            time = np.array([0, 1])
        nbin = np.array([
            args[0].size-1, args[1].size-1, args[2].size-1, args[3].size-1,
            args[4].size-1, time.size-1, 1
            ], dtype="u8")
        binmin = np.array([
            args[0][0], args[1][0], args[2][0], args[3][0],
            args[4][0], time[0], charge[0] - 1])
        binmax = np.array([
            args[0][-1], args[1][-1], args[2][-1], args[3][-1],
            args[4][-1], time[-1], charge[0] + 1])
        if(exi):
            binmin[3] *= unyt.elementary_charge
            binmax[3] *= unyt.elementary_charge
            coordinates = np.array([0, 1, 2, 10, 11, 14, 15], dtype="uint32")
        else:
            coordinates = np.array([0, 1, 2, 5, 6, 14, 15], dtype="uint32")
        if(toroidal):
            coordinates[0] = 3
            coordinates[1] = 4
            coordinates[2] = 1
        _LIBASCOT.hist_init(
            ctypes.byref(prod),
            coordinates.size,
            coordinates.ctypes.data_as(ctypes.POINTER(ctypes.c_uint)),
            binmin.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            binmax.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            nbin.ctypes.data_as(ctypes.POINTER(ctypes.c_size_t))
            )
        return prod

    def _build_distdata(self, prod):
        """
        Transform the raw histogram data into a DistData object.

        Parameters
        ----------
        prod : STRUCT_HIST
            Histogram structure containing the raw data.
        """
        # We get the numpy representation of the data.
        data = np.array(prod.bins[0:prod.nbin], dtype='f8')

        # Then we get the sizes.
        sizes = []
        names = []
        boundaries = []
        units = {'r': unyt.m, 'phi': unyt.rad, 'z': unyt.m,
                 'rho': unyt.dimensionless, 'theta': unyt.rad,
                 'ppar': unyt.kg * unyt.m / unyt.s,
                 'pperp': unyt.kg * unyt.m / unyt.s,
                 'ekin': unyt.J, 'xi': unyt.dimensionless,
                 'pr': unyt.kg * unyt.m / unyt.s,
                 'pz': unyt.kg * unyt.m / unyt.s,
                 'pphi': unyt.kg * unyt.m / unyt.s,
                 'mu': unyt.J / unyt.T,
                 'ptor': unyt.C * unyt.Wb,
                 'time': unyt.s, 'charge': unyt.e}
        for i in range(len(hist_coordinate__enumvalues)):
            if prod.axes[i].n > 0:
                sizes.append(prod.axes[i].n)
                names.append(hist_coordinate__enumvalues[i].lower())
                edges = np.linspace(
                    prod.axes[i].min, prod.axes[i].max, prod.axes[i].n + 1)
                boundaries.append(edges)
        
        abscissae = dict()
        for ii in range(len(sizes)):
            abscissae[names[ii]] = boundaries[ii] * units[names[ii]]
        
        data = data.reshape(sizes, order='C')
        return DistData(data, **abscissae)

    def reactions(self, reaction=None):
        """Return reaction data for a given reaction.

        Parameters
        ----------
        reaction : str, optional
            Reaction or None to return list of available reactions.

        Returns
        -------
        reactions : [str]
            List of available reactions if ``reaction=None``.
        m1 : float
            Mass of reactant 1.
        q1 : float
            Charge of reactant 1.
        m2 : float
            Mass of reactant 2.
        q2 : float
            Charge of reactant 2.
        mprod1 : float
            Mass of product 1.
        qprod1 : float
            Charge of product 1.
        mprod2 : float
            Mass of product 2.
        qprod2 : float
            Charge of product 2.
        q : float
            Energy released in the reaction.
        """
        reactions = {v: k for k, v in AFSI_REACTIONS.items()}
        if reaction is None:
            return reactions.keys()
        if not reaction in reactions:
            raise ValueError("Unknown reaction")

        m1     = np.zeros((1,), dtype="f8")
        q1     = np.zeros((1,), dtype="f8")
        m2     = np.zeros((1,), dtype="f8")
        q2     = np.zeros((1,), dtype="f8")
        mprod1 = np.zeros((1,), dtype="f8")
        qprod1 = np.zeros((1,), dtype="f8")
        mprod2 = np.zeros((1,), dtype="f8")
        qprod2 = np.zeros((1,), dtype="f8")
        q      = np.zeros((1,), dtype="f8")
        fun = _LIBASCOT.boschhale_reaction
        fun.restype  = None
        fun.argtypes = [ctypes.c_int, PTR_REAL, PTR_REAL, PTR_REAL, PTR_REAL,
                        PTR_REAL, PTR_REAL, PTR_REAL, PTR_REAL, PTR_REAL]
        fun(reactions[reaction], m1, q1, m2, q2, mprod1, qprod1, mprod2,
            qprod2, q)

        return m1*unyt.kg, q1*unyt.C, m2*unyt.kg, q2*unyt.C, mprod1*unyt.kg, \
            qprod1*unyt.C, mprod2*unyt.kg, qprod2*unyt.C, q*unyt.eV

class AfsiMixin(DistMixin):
    """Mixin class with post-processing results related to AFSI.
    """

    def _require(self, *args):
        """Check if required data is present and raise exception if not.

        This is a helper function to quickly check that the data is available.

        Parameters
        ----------
        *args : `str`
            Name(s) of the required data.

        Raises
        ------
        AscotNoDataException
            Raised if the required data is not present.
        """
        for arg in args:
            if not hasattr(self, arg):
                raise AscotNoDataException(
                    "Data for \"" +  arg + "\" is required but not present.")

    def getdist(self, dist, exi=False, ekin_edges=None, pitch_edges=None,
                plotexi=False):
        """Return 5D distribution function of one of the fusion products.

        Parameters
        ----------
        dist : {"prod1", "prod2"}
            Which product to return.
        exi : bool, optional
            Convert the momentum space to energy-pitch.

            The distribution is normalized to conserve the particle number.
        ekin_edges : int or array_like, optional
            Number of bins or bin edges in the energy abscissa if ``exi=True``.
        pitch_edges : int or array_like, optional
            Number of bins or bin edges in the pitch abscissa if ``exi=True``.
        plotexi : bool, optional
            Visualize the transformation from ppar-perp to energy-pitch if
            if ``exi=True``.

            Use this option to adjust energy and pitch abscissae to your liking.

        Returns
        -------
        data : :class:`DistData`
            The distribution data object.
        """
        if dist == "prod1":
            self._require("_prod1dist5d", "_reaction")
            distout = self._prod1dist5d.get()
            mass = self._reaction.get()[2]
        elif dist == "prod2":
            self._require("_prod2dist5d", "_reaction")
            distout = self._prod2dist5d.get()
            mass = self._reaction.get()[3]
        else:
            raise ValueError("dist must be either 'prod1' or 'prod2'")

        return self._getdist(distout, mass, exi=exi, ekin_edges=ekin_edges,
                             pitch_edges=pitch_edges, plotexi=plotexi)
