import json
from json import JSONEncoder
from seer.automation.autofix.event_manager import AutofixStatus

class CustomJSONEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, AutofixStatus):
            return obj.value
        return super().default(obj)