def extractParams(param_str):
    """Parse a semicolon-separated 'key: value' parameter string into a dict.

    Example:
        "Km_C00001: 0.5; Kcat_F: 10.0" -> {"Km_C00001": "0.5", "Kcat_F": "10.0"}

    KI entries (e.g. "C00001_KI: 0.1") are renamed to "Ki_C00001".
    """
    if not isinstance(param_str, str) or not param_str.strip():
        return {}
    if ',' in param_str:
        raise NameError('Comma is an invalid character in a parameter string.')
    if ':' not in param_str:
        raise NameError('No key:value pairs found in parameter string.')

    mydict = {}
    for s in param_str.split(';'):
        s = s.strip()
        if not s:
            continue
        k, v = s.split(':', 1)
        k = k.strip()
        v = v.strip()
        if 'KI' in k:
            parts = k.split('_')
            k = parts[1].replace('I', 'i') + '_' + parts[0]
        mydict[k] = v
    return mydict


FIND = ['-', ',', '+', ' ']
REPLACE = ['_', '_', 'plus', '']


def fmt(x):
    """Format a species name to be Antimony-compatible (no special characters)."""
    x = str(x)
    try:
        if getStoich(x)[0] and getStoich(x)[0][-1].isnumeric() and len(getStoich(x)[1]) > 0 and getStoich(x)[1][0] != ' ':
            x = 'z' + x
    except Exception:
        pass
    for f, r in zip(FIND, REPLACE):
        x = x.replace(f, r)
    return x.strip()


def getStoich(sp):
    """Split a stoichiometry string into (coefficient, species_name)."""
    i = 0
    if len(sp) > 0:
        while i < len(sp) and sp[i].isnumeric():
            i += 1
    return sp[:i], sp[i:]
