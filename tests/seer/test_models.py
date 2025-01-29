import pytest
from pydantic import ValidationError

from seer.automation.models import StacktraceFrame, Stacktrace


def test_stacktrace_frame_validation_with_missing_in_app():
    """Test that StacktraceFrame handles missing inApp field correctly"""
    # This should now work after the fix
    frame = {
        "absPath": "src/file.py",
        "filename": "file.py",
        "function": "test_func",
        "lineNo": 42,
        "colNo": 10,
        # inApp is intentionally missing
    }
    
    frame_obj = StacktraceFrame.model_validate(frame)
    assert frame_obj.in_app is False  # Verify default value


def test_stacktrace_frame_validation_with_none_in_app():
    """Test that StacktraceFrame handles None inApp value correctly"""
    frame = {
        "absPath": "src/file.py",
        "filename": "file.py",
        "function": "test_func",
        "lineNo": 42,
        "colNo": 10,
        "inApp": None,
    }
    
    frame_obj = StacktraceFrame.model_validate(frame)
    assert frame_obj.in_app is False  # Verify default value


def test_stacktrace_frame_validation_with_valid_in_app():
    """Test that StacktraceFrame handles valid boolean inApp values"""
    frame = {
        "absPath": "src/file.py",
        "filename": "file.py",
        "function": "test_func",
        "lineNo": 42,
        "colNo": 10,
        "inApp": True,
    }
    
    frame_obj = StacktraceFrame.model_validate(frame)
    assert frame_obj.in_app is True


def test_stacktrace_validation_with_mixed_frames():
    """Test that Stacktrace can handle a mix of frame data types"""
    frames = [
        # Frame with missing inApp
        {
            "absPath": "src/file1.py",
            "filename": "file1.py", 
            "function": "func1",
            "lineNo": 42
        },
        # Frame with None inApp
        {
            "absPath": "src/file2.py",
            "filename": "file2.py",
            "function": "func2", 
            "lineNo": 84,
            "inApp": None
        },
        # Frame with valid inApp
        {
            "absPath": "src/file3.py", 
            "filename": "file3.py",
            "function": "func3",
            "lineNo": 126,
            "inApp": True
        }
    ]
    
    stacktrace = Stacktrace(frames=frames)
    assert len(stacktrace.frames) == 3
    assert stacktrace.frames[0].in_app is False  # Default for missing
    assert stacktrace.frames[1].in_app is False  # None converted to False
    assert stacktrace.frames[2].in_app is True   # Preserved True value