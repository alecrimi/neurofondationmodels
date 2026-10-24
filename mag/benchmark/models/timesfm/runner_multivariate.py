"""TimesFM multivariate adapter scaffold, isolated from EEG preprocessing.

Input: trusted pickle with contexts only; context shape is (C,T).
Output: subject -> {predictions: float array(W,C,H)} in input signal units.

Implementation decision intentionally left open:
* the project's current TimesFM 2.5 is univariate with an XReg extension;
* native joint channel forecasting requires a different supported backend;
* batching independent series does not constitute multivariate forecasting.

For XReg, explicitly define how future covariates are obtained. The common
input contains only observed history and must not be extended with hidden
future EEG targets as if they were available at forecast time.

Do not infer a backend from installed package version or silently change
checkpoint. Implement load_model and predict_joint after selecting a protocol.
"""
from __future__ import annotations
import argparse
import pickle
import numpy as np


def load_model(device: str):
    """Load the explicitly selected TimesFM multivariate backend once.

    API class, checkpoint, dependency versions and device placement belong
    here. They are intentionally not fixed in this scaffold.
    """
    raise NotImplementedError(
        'TimesFM: select native multivariate inference or an explicit XReg protocol '
        'and implement load_model/predict_joint; current univariate runner is unchanged.'
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
    raise NotImplementedError('TimesFM multivariate prediction API is not implemented.')


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
