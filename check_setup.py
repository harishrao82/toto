
try:
    from toto.model.toto import Toto
    from toto.inference.forecaster import TotoForecaster
    from toto.data.util.dataset import MaskedTimeseries
    
    # Load the model
    toto = Toto.from_pretrained('Datadog/Toto-Open-Base-1.0')
    print("✓ Toto-TS installed successfully!")
except ImportError as e:
    print(f"✗ Installation issue: {e}")