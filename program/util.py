def str_to_bool(string):
    if not string:
        return False
    string_lower = string.lower()
    if string_lower in ['true', 't', '1', 'yes', 'y', 'on']:
        return True
    if string_lower in ['false', 'f', '0', 'no', 'n', 'off']:
        return False
    raise ValueError('string(%s) cannot convert to boolean.' % string)
