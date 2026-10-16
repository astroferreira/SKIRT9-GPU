"""Explicit on-demand dust choices, included in export and render identities."""


def physical_options(entry):
    options = dict(diffuse_dust='torrey', diffuse_grain_mix='dl07')
    supplied = entry.get('physical_options', {})
    if set(supplied) - set(options):
        raise ValueError('Unknown on-demand physical option')
    options.update(supplied)
    if options['diffuse_dust'] not in ('torrey', 'rr2014'):
        raise ValueError('Unknown diffuse dust prescription')
    if options['diffuse_grain_mix'] not in ('dl07', 'wd01'):
        raise ValueError('Unknown diffuse grain mixture')
    return options


def dust_description(options):
    mass = 'Torrey ISM, dust-to-metal 0.2' if options['diffuse_dust'] == 'torrey' else 'RR2014, no whole-cell cloud subtraction'
    grains = 'DraineLiDustMix' if options['diffuse_grain_mix'] == 'dl07' else 'WeingartnerDraineDustMix (MilkyWay)'
    return f'{mass}, {grains}, stochastic'
