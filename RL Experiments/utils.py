import confs

def get_exp_conf(exp: str)->dict:
    return getattr(confs, exp)()