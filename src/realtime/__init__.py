from src.realtime.streaming import StreamingDebateSession

try:
    from src.realtime.voice_pipeline import VoiceDebatePipeline
    __all__ = ["VoiceDebatePipeline", "StreamingDebateSession"]
except ImportError:
    __all__ = ["StreamingDebateSession"]
