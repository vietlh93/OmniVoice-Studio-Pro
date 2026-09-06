"""
Model lifecycle helpers for OmniVoice Gradio app.
"""

from OmniVoice.omnivoice_inference.ttsOmni import Omni

DEVICE = "cpu"
OMNI_MODEL = None


def configure_device(device: str) -> None:
    """Initialize runtime device from main app."""
    global DEVICE
    DEVICE = device


def get_omni_model():
    """Load Omni model one time and reuse."""
    global OMNI_MODEL
    if OMNI_MODEL is None:
        print("=" * 50)
        print("🚀 Loading OmniVoice...")
        print("=" * 50)
        OMNI_MODEL = Omni()
        OMNI_MODEL.loadOmniFromUI()
        print("✅ OmniVoice loaded!")
        print("=" * 50)
    return OMNI_MODEL