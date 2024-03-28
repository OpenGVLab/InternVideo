import itertools

def _block_set(ia_blocks):
    if len(ia_blocks) > 0 and isinstance(ia_blocks[0], list):
        ia_blocks = list(itertools.chain.from_iterable(ia_blocks))
    return ia_blocks

def has_person(ia_config):
    ia_blocks = _block_set(ia_config.I_BLOCK_LIST)
    return (ia_config.ACTIVE and 'P' in ia_blocks and ia_config.MAX_PERSON > 0)


def has_object(ia_config):
    ia_blocks = _block_set(ia_config.I_BLOCK_LIST)
    return (ia_config.ACTIVE and 'O' in ia_blocks and ia_config.MAX_OBJECT > 0)


def has_memory(ia_config):
    ia_blocks = _block_set(ia_config.I_BLOCK_LIST)
    return (ia_config.ACTIVE and 'M' in ia_blocks and ia_config.MAX_PER_SEC > 0)
