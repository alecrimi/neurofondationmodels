"""Future Chronos-2 multivariate adapter scaffold, isolated from EEG preprocessing.

Input: trusted pickle with contexts only; context shape is (C,T).
Output: subject -> {predictions: float array(W,C,H)} in input signal units.

No Chronos/Bolt imports, checkpoint names, quantile indices or backend
argument names are fixed here. Implement only load_model and predict_joint
when the project migrates to the selected native Chronos-2 backend.
The current Chronos-Bolt univariate runner remains separate.
Do not fall back to independent univariate forecasts under this label.
"""
from __future__ import annotations
import argparse
import pickle
import numpy as np


def load_model(device: str):
    """Load the explicitly selected Chronos-2 multivariate backend once.

    API class, checkpoint, dependency versions and device placement belong
    here. They are intentionally not fixed in this scaffold.
    """
    raise NotImplementedError(
        'Chronos-2: implement load_model/predict_joint after the backend migration; '
        'the current Chronos-Bolt univariate runner is unchanged.'
    )


def predict_joint(model, context: np.ndarray, horizon_len: int, channels: list[str]):
    """Forecast all channels jointly using only historical context.

    Parameters
    ----------
    model : object
        Loaded backend.
    context : np.ndarray
        Historical signals, shape (C,T), ordered as channels.
    horizon_len : int
        Number of future samples per channel.
    channels : list[str]
        Ordered channel/parcel identifiers.

    Returns
    -------
    np.ndarray
        Point forecasts with shape (C,H), in the same scale as context.
        Convert backend dimensions and choose the point estimate here.
    """
    raise NotImplementedError('Chronos-2 multivariate prediction API is not implemented.')


def main(argv=None):
    """Read the common inference payload and write aligned joint forecasts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    with open(args.input, 'rb') as stream:
        data = pickle.load(stream)
    if data.get('schema_version') != 1:
        raise ValueError('Unsupported multivariate input schema.')
    model = load_model(data['device'])
    result = {}
    for sid, subject in data['subjects'].items():
        forecasts = []
        for window in subject['windows']:
            pred = np.asarray(predict_joint(model, window['context'],
                                           data['horizon_len'], data['channels']))
            expected = (len(data['channels']), data['horizon_len'])
            if pred.shape != expected or not np.isfinite(pred).all():
                raise ValueError(f'{sid}: invalid joint forecast; expected {expected}, got {pred.shape}.')
            forecasts.append(pred)
        result[sid] = {'predictions': np.stack(forecasts)}
    with open(args.output, 'wb') as stream:
        pickle.dump(result, stream)


if __name__ == '__main__':
    main()
