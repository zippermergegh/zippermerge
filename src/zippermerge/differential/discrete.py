def diff(t, x):
    """
    Computes the discrete derivative of timeseries data.
    
    Parameters:
    t (list): Python array containing time values t_k.
    x (list): Python array containing signal values x(t_k).
    
    Returns:
    list: The computed discrete derivative v(t) as a Python array.
    """
    
    # Check for equality of lengths of input arrays
    if len(t) != len(x):
        raise ValueError("The time array (t) and signal array (x) must be of equal length.")
    
    v = []
    # Iterate through arrays starting from second element
    for k in range(1, len(t)):
        # Calculate difference between consecutive signal values
        delta_x = x[k] - x[k-1]
        
        # Calculate difference between consecutive time values
        delta_t = t[k] - t[k-1]
        
        # Compute discrete derivative 
        v_t = delta_x / delta_t
        
        # Append the value to the v array
        v.append(v_t)
        
    return v
