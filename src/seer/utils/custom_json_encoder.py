import json
from src.seer.automation.autofix.event_manager import AutofixStatus

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, AutofixStatus):
            return obj.value
        return super().default(obj)