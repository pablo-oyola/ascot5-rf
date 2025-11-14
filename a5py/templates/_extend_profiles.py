import numpy as np
import xarray as xr
from scipy.interpolate import make_splrep
import lmfit
import warnings

def mtanh(x, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9):
        """
        Modified-tanh used for profile fitting.

        Parameters
        ----------
        x : np.ndarray
            The radial grid.
        a0: float
            Value at the SOL.
        a1: float
            Value at the pedestal top.
        a2: float
            Position of the half-width of the pedestal.
        a3: float
            Half-width of the pedestal.
        a4, a5, a6: float
            Parameters to control the core polynomial.
        a7, a8, a9: float
            Parameters to control the edge polynomial.
        """
        diff_a = (a1 - a0)/2
        sum_a = (a1 + a0)/2
        
        core_polynomial = (1 + a4 * x + a5 * x**2 + a6 * x**3)
        edge_polynomial = (1 + a7 * x + a8 * x**2 + a9 * x**3)
        z = (a2 - x) / a3
        expz = np.exp(z)
        expmz = np.exp(-z)
        denominator = expz + expmz

        return sum_a + diff_a * (core_polynomial * expz - edge_polynomial * expmz) / denominator

def tanh_fit(x, amp, offset, center, width):
    """
    A simple tanh function used for profile fitting.

    Parameters
    ----------
    x : np.ndarray
        The radial grid.
    amp : float
        The amplitude of the tanh function.
    offset : float
        The offset of the tanh function.
    center : float
        The center of the tanh function.
    width : float
        The width of the tanh function.
    -----------
    Returns
    -------
    np.ndarray
        The value of the tanh function at the given radial grid.
    -----------
    """
    return amp * (1.0 - np.tanh((x - center) / width)) + offset

def mtanh_temp(x: float, a0: float, a1: float, a2: float, a3: float, a4: float):
    """
    A modified-tanh function used for temperature profile fitting.

    Parameters
    ----------
    x : float
        The radial grid.
    a0 : float
        The amplitude of the tanh function.
    a1 : float
        Center of the tanh function.
    a2 : float
        Width of the tanh function.
    a3, a4 : float
        Polynomial coefficients.
    """
    tanh_part = a0 * (1.0 - np.tanh((x - a1) / a2))
    polynomial_part = 1 + a3 * x**1.5 + a4 * x**2
    return tanh_part * polynomial_part

def get_physical_meaning(result: lmfit.model.ModelResult, model: str) -> dict:
    """
    Extract the physical meaning of the fitted parameters from the result.

    Parameters
    ----------
    result : lmfit.model.ModelResult
        The result of the fitting.
    model : str
        The model used for fitting. It can be 'mtanh', 'tanh', or 'mtanh_temp'.

    Returns
    -------
    dict
        A dictionary with the physical meaning of the fitted parameters.
    """
    params = result.params
    if model.lower() == 'tanh':
        return {
            'height': params['amp'].value,
            'offset': params['offset'].value,
            'center': params['center'].value,
            'width': params['width'].value
        }
    elif model.lower() == 'mtanh':
        return {
            'height': params['a1'].value,
            'offset': params['a0'].value,
            'center': params['a2'].value,
            'width': params['a3'].value,
            'core_polynomial': (params['a4'].value, params['a5'].value, params['a6'].value),
            'edge_polynomial': (params['a7'].value, params['a8'].value, params['a9'].value)
        }
    elif model.lower() == 'mtanh_temp':
        return {
            'height': params['a0'].value,
            'center': params['a1'].value,
            'width': params['a2'].value,
            'offset': 0.0,
            'polynomial': (params['a3'].value, params['a4'].value)
        }

def prepare_models(model: str, input_prof: float,
                   fix_ped_width: float=None, 
                   fix_ped_pos: float=None):
    """
    Return the model from LMFIT for the input model name.
    """
    if model.lower() not in ['mtanh', 'tanh', 'mtanh_temp']:
        raise ValueError(f"Model {model} not recognized. "
                         "Available models: 'mtanh', 'tanh', 'mtanh_temp'.")
    
    # Preparing each of the models with the corresponding hints.
    if model.lower() == 'tanh':
        model = lmfit.Model(tanh_fit)

        lower_bounds = {'amp': 0.0, 
                        'offset': 0.0, 
                        'center': 0.0, 
                        'width': 1e-5}
        upper_bounds = {'amp': np.inf, 
                        'offset': 5.0, 
                        'center': 1.0, 
                        'width': np.inf}
        initial_guess = {'amp': input_prof.max(), 
                         'offset': input_prof.min(), 
                         'center': 1.0, 'width': 0.1}

        if fix_ped_width is not None:
            lower_bounds['width'] = fix_ped_width
            upper_bounds['width'] = fix_ped_width
            initial_guess['width'] = fix_ped_width
        if fix_ped_pos is not None:
            lower_bounds['center'] = fix_ped_pos
            upper_bounds['center'] = fix_ped_pos
            initial_guess['center'] = fix_ped_pos

        for ii in initial_guess:
            if lower_bounds[ii] == upper_bounds[ii]:
                vary = False
            else:
                vary = True
            model.set_param_hint(ii, value=initial_guess[ii],
                                 min=lower_bounds[ii],
                                 max=upper_bounds[ii],
                                 vary=vary)
    elif model.lower() == 'mtanh':
        model = lmfit.Model(mtanh)

        lower_bounds = {'a0': 0.0, 
                        'a1': 0.0, 
                        'a2': 0.0, 'a3': 1e-5,
                        'a4': -np.inf, 'a5': -np.inf, 'a6': -np.inf,
                        'a7': -np.inf, 'a8': -np.inf, 'a9': -np.inf}
        upper_bounds = {'a0': np.inf, 'a1': 5.0, 'a2': 1.0, 'a3': np.inf,
                        'a4': np.inf, 'a5': np.inf, 'a6': np.inf,
                        'a7': np.inf, 'a8': np.inf, 'a9': np.inf}
        initial_guess = {'a0': input_prof.max(), 'a1': input_prof.min(), 
                         'a2': 1.0, 'a3': 0.05,
                         'a4': 0.0, 'a5': 0.0, 'a6': 0.0,
                         'a7': 1.0, 'a8': 1.0, 'a9': 1.0}

        if fix_ped_width is not None:
            print("Fixing pedestal width to", fix_ped_pos)
            lower_bounds['a3'] = fix_ped_width
            upper_bounds['a3'] = fix_ped_width
            initial_guess['a3'] = fix_ped_width
        if fix_ped_pos is not None:
            print("Fixing pedestal position to", fix_ped_pos)
            lower_bounds['a2'] = fix_ped_pos
            upper_bounds['a2'] = fix_ped_pos
            initial_guess['a2'] = fix_ped_pos

        for ii in initial_guess:
            model.set_param_hint(ii, value=initial_guess[ii],
                                 min=lower_bounds[ii],
                                 max=upper_bounds[ii])
    elif model.lower() == 'mtanh_temp':
        model = lmfit.Model(mtanh_temp)

        lower_bounds = {'a0': 0.0, 'a1': 0.0, 'a2': 1e-5, 'a3': -np.inf, 'a4': -np.inf}
        upper_bounds = {'a0': np.inf, 'a1': 5.0, 'a2': np.inf, 'a3': np.inf, 'a4': np.inf}
        initial_guess = {'a0': 1.0, 'a1': 0.5, 'a2': 0.1, 'a3': 0.0, 'a4': 0.0}

        if fix_ped_width is not None:
            lower_bounds['a2'] = fix_ped_width
            upper_bounds['a2'] = fix_ped_width
            initial_guess['a2'] = fix_ped_width
        if fix_ped_pos is not None:
            lower_bounds['a1'] = fix_ped_pos
            upper_bounds['a1'] = fix_ped_pos
            initial_guess['a1'] = fix_ped_pos

        for ii in initial_guess:
            model.set_param_hint(ii, value=initial_guess[ii],
                                 min=lower_bounds[ii],
                                 max=upper_bounds[ii])
    else:
        raise ValueError(f"Model {model} not recognized. "
                         "Available models: 'mtanh', 'tanh', 'mtanh_temp'.")
    return model

def extend_profile(rhop: float, profile: float,
                   fitting_region: float = [0.9, 1.0],
                   fix_ped_width: float=None,
                   fix_ped_pos: float=None,
                   smoothing_joint: float=0.01,
                   allow_negative: bool=False,
                   modelname: str='mtanh', mute: bool=False) -> xr.DataArray:
    """
    Extends a given profile to cover the range of rhopol beyond
    the separatrix.

    Parameters
    ----------
    rhop : float
        The radial grid where the profile is defined. Usually 
        this is cropped at rhopol=1 and that is why you need this
        function.
    profile : float
        The profile to be extended.
    fix_ped_width : float, optional
        If provided, the width of the pedestal will be fixed to this value.
        This is useful for cases where you want to maintain a specific pedestal width.
    fix_ped_pos : float, optional
        If provided, the position of the pedestal will be fixed to this value.
        This is useful for cases where you want to maintain a specific pedestal position.
    smoothing_joint : float, optional
        The smoothing factor for the joint region. This is the smoothing factor
        used to blend the extended profile with the original profile
    allow_negative : bool, optional
        If True, allows the extended profile to have negative values.
    """
    # Checking the inputs.
    if fix_ped_width is not None and fix_ped_width <= 0:
        raise ValueError("fix_ped_width must be a positive value.")
    if fix_ped_pos is not None and (fix_ped_pos < 0 or fix_ped_pos > 1):
        raise ValueError("fix_ped_pos must be between 0 and 1.")
    
    # Ensure profile is a numpy array
    profile = np.asarray(profile)
    rhop = np.asarray(rhop)

    # We take the region where the profile is defined to fit it.
    mask = (rhop >= fitting_region[0]) & (rhop <= fitting_region[1])
    rhop_fit = rhop[mask]
    profile_fit = profile[mask]
    
    # Initial guess for the parameters
    model = prepare_models(modelname, profile_fit, fix_ped_width, fix_ped_pos)

    # Fitting.
    result = model.fit(profile_fit, x=rhop_fit)
    if not mute:
        print("Fitting result:"
            f"\n{result.fit_report()}")

    # --- Merging the extended profile with the original profile ---
    # Create a new radial grid that extends beyond the original profile
    rhop_extended = np.linspace(fitting_region[0], 2.0, 1000)
    # Calculate the extended profile using the fitted parameters
    extended_profile = result.eval(x=rhop_extended)
    # Smooth the joint region
    flags = rhop < fitting_region[0]
    rhop4smooth = np.concatenate((rhop[flags], rhop_extended))
    profile4smooth = np.concatenate((profile[flags], extended_profile))
    if mute:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spline = make_splrep(rhop4smooth, profile4smooth, s=smoothing_joint)
    else:
        spline = make_splrep(rhop4smooth, profile4smooth, s=smoothing_joint)
    # Evaluate the spline on the extended radial grid

    rhopol_out = np.linspace(0, 2.0, 1000)
    extended_profile_smooth = spline(rhopol_out)

    # Ensure the extended profile is non-negative if required
    if not allow_negative:
        extended_profile_smooth = np.clip(extended_profile_smooth, 
                                          extended_profile_smooth.max()*1e-10, 
                                          None)
    
    # Create an xarray DataArray for the extended profile
    extended_profile_da = xr.DataArray(
        extended_profile_smooth,
        coords=[rhopol_out],
        dims=["rhopol"],
        attrs={
            "description": "Extended profile",
            "allow_negative": allow_negative
        }
    )
    extended_profile_da.attrs.update(get_physical_meaning(result, modelname))

    return extended_profile_da