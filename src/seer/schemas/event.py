import typing

#  The sentry v7 event structure.
Event = typing.TypedDict(
    "Event",
    {
        "breadcrumbs": typing.Mapping[str, typing.Any],
        "contexts": typing.Union["Contexts", None],
        "culprit": str,
        "debug_meta": typing.Union["DebugMeta", None],
        "dist": str,
        "environment": str,
        "errors": typing.List[typing.Union["EventProcessingError", None]],
        "event_id": typing.Union["EventId", None],
        "exception": typing.Mapping[str, typing.Any],
        "extra": typing.Mapping[str, typing.Any],
        "fingerprint": typing.Union["Fingerprint", None],
        "level": typing.Union["Level", None],
        "logentry": typing.Union["LogEntry", None],
        "logger": str,
        "modules": typing.Mapping[str, typing.Any],
        "platform": str,
        "received": typing.Union["Timestamp", None],
        "release": str,
        "request": typing.Union["Request", None],
        "sdk": typing.Union["ClientSdkInfo", None],
        "server_name": str,
        "stacktrace": typing.Union["Stacktrace", None],
        "tags": typing.Union["Tags", None],
        "threads": typing.Mapping[str, typing.Any],
        # format: uint64
        "time_spent": int,
        "timestamp": typing.Union["Timestamp", None],
        "transaction": str,
        "transaction_info": typing.Union["TransactionInfo", None],
        "type": typing.Union["EventType", None],
        "user": typing.Union["User", None],
        "version": str,
    },
)

Addr = str

#  Application information.
#
#  App context describes the application. As opposed to the runtime, this is the actual
#  application that was running and carries metadata about the current session.
AppContext = typing.TypedDict(
    "AppContext",
    {
        "app_build": str,
        "app_identifier": str,
        # format: uint64
        "app_memory": int,
        "app_name": str,
        "app_start_time": str,
        "app_version": str,
        "build_type": str,
        "device_app_hash": str,
        "in_foreground": bool,
        "view_names": typing.List[str],
    },
)

#  Legacy apple debug images (MachO).
#
#  This was also used for non-apple platforms with similar debug setups.
AppleDebugImage = typing.TypedDict(
    "AppleDebugImage",
    {
        "arch": str,
        # format: uint64
        "cpu_subtype": int,
        # format: uint64
        "cpu_type": int,
        "image_addr": typing.Union["Addr", None],
        # format: uint64
        "image_size": int,
        "image_vmaddr": typing.Union["Addr", None],
        "name": str,
        # format: uuid
        "uuid": str,
    },
)

#  The Breadcrumbs Interface specifies a series of application events, or "breadcrumbs", that
#  occurred before an event.
#
#  An event may contain one or more breadcrumbs in an attribute named `breadcrumbs`. The entries
#  are ordered from oldest to newest. Consequently, the last entry in the list should be the last
#  entry before the event occurred.
#
#  While breadcrumb attributes are not strictly validated in Sentry, a breadcrumb is most useful
#  when it includes at least a `timestamp` and `type`, `category` or `message`. The rendering of
#  breadcrumbs in Sentry depends on what is provided.
#
#  The following example illustrates the breadcrumbs part of the event payload and omits other
#  attributes for simplicity.
#
#  ```json
#  {
#    "breadcrumbs": {
#      "values": [
#        {
#          "timestamp": "2016-04-20T20:55:53.845Z",
#          "message": "Something happened",
#          "category": "log",
#          "data": {
#            "foo": "bar",
#            "blub": "blah"
#          }
#        },
#        {
#          "timestamp": "2016-04-20T20:55:53.847Z",
#          "type": "navigation",
#          "data": {
#            "from": "/login",
#            "to": "/dashboard"
#          }
#        }
#      ]
#    }
#  }
#  ```
Breadcrumb = typing.TypedDict(
    "Breadcrumb",
    {
        "category": str,
        "data": typing.Mapping[str, typing.Any],
        "event_id": typing.Union["EventId", None],
        "level": typing.Union["Level", None],
        "message": str,
        "timestamp": typing.Union["Timestamp", None],
        "type": str,
    },
)

#  Web browser information.
BrowserContext = typing.TypedDict(
    "BrowserContext",
    {
        "name": str,
        "version": str,
    },
)

#  POSIX signal with optional extended data.
#
#  Error codes set by Linux system calls and some library functions as specified in ISO C99,
#  POSIX.1-2001, and POSIX.1-2008. See
#  [`errno(3)`](https://man7.org/linux/man-pages/man3/errno.3.html) for more information.
CError = typing.TypedDict(
    "CError",
    {
        "name": str,
        # format: int64
        "number": int,
    },
)

#  The SDK Interface describes the Sentry SDK and its configuration used to capture and transmit an event.
ClientSdkInfo = typing.TypedDict(
    "ClientSdkInfo",
    {
        "integrations": typing.List[str],
        "name": str,
        "packages": typing.List[typing.Union["ClientSdkPackage", None]],
        "version": str,
    },
)

#  An installed and loaded package as part of the Sentry SDK.
ClientSdkPackage = typing.TypedDict(
    "ClientSdkPackage",
    {
        "name": str,
        "version": str,
    },
)

#  Cloud Resource Context.
#
#  This context describes the cloud resource the event originated from.
#
#  Example:
#
#  ```json
#  "cloud_resource": {
#      "cloud.account.id": "499517922981",
#      "cloud.provider": "aws",
#      "cloud.platform": "aws_ec2",
#      "cloud.region": "us-east-1",
#      "cloud.vavailability_zone": "us-east-1e",
#      "host.id": "i-07d3301208fe0a55a",
#      "host.type": "t2.large"
#  }
#  ```
CloudResourceContext = typing.TypedDict(
    "CloudResourceContext",
    {
        "cloud.account.id": str,
        "cloud.availability_zone": str,
        "cloud.platform": str,
        "cloud.provider": str,
        "cloud.region": str,
        "host.id": str,
        "host.type": str,
    },
)

CodeId = str

#  A context describes environment info (e.g. device, os or browser).
Context = typing.Union[
    "DeviceContext",
    "OsContext",
    "RuntimeContext",
    "AppContext",
    "BrowserContext",
    "GpuContext",
    "TraceContext",
    "ProfileContext",
    "ReplayContext",
    "UserReportV2Context",
    "MonitorContext",
    "ResponseContext",
    "OtelContext",
    "CloudResourceContext",
    "NelContext",
    typing.Mapping[str, typing.Any],
]

ContextInner = typing.Any

#  The Contexts interface provides additional context data. Typically, this is data related to the
#  current user and the environment. For example, the device or application version. Its canonical
#  name is `contexts`.
#
#  The `contexts` type can be used to define arbitrary contextual data on the event. It accepts an
#  object of key/value pairs. The key is the “alias” of the context and can be freely chosen.
#  However, as per policy, it should match the type of the context unless there are two values for
#  a type. You can omit `type` if the key name is the type.
#
#  Unknown data for the contexts is rendered as a key/value list.
#
#  For more details about sending additional data with your event, see the [full documentation on
#  Additional Data](https://docs.sentry.io/enriching-error-data/additional-data/).
Contexts = typing.Mapping[str, typing.Union["ContextInner", None]]

#  A map holding cookies.
Cookies = typing.Any

#  The arbitrary data on the trace.
Data = typing.TypedDict(
    "Data",
    {
        "previousRoute": typing.Union["Route", None],
        "route": typing.Union["Route", None],
    },
)

DebugId = str

#  A debug information file (debug image).
DebugImage = typing.Union[
    "AppleDebugImage",
    "NativeDebugImage",
    "NativeDebugImage",
    "NativeDebugImage",
    "NativeDebugImage",
    "NativeDebugImage",
    "ProguardDebugImage",
    "NativeDebugImage",
    "SourceMapDebugImage",
    "JvmDebugImage",
    typing.Mapping[str, typing.Any],
]

#  Debugging and processing meta information.
#
#  The debug meta interface carries debug information for processing errors and crash reports.
#  Sentry amends the information in this interface.
#
#  Example (look at field types to see more detail):
#
#  ```json
#  {
#    "debug_meta": {
#      "images": [],
#      "sdk_info": {
#        "sdk_name": "iOS",
#        "version_major": 10,
#        "version_minor": 3,
#        "version_patchlevel": 0
#      }
#    }
#  }
#  ```
DebugMeta = typing.TypedDict(
    "DebugMeta",
    {
        "images": typing.List[typing.Union["DebugImage", None]],
        "sdk_info": typing.Union["SystemSdkInfo", None],
    },
)

#  Device information.
#
#  Device context describes the device that caused the event. This is most appropriate for mobile
#  applications.
DeviceContext = typing.TypedDict(
    "DeviceContext",
    {
        "arch": str,
        # format: double
        "battery_level": float,
        "battery_status": str,
        "boot_time": str,
        "brand": str,
        "charging": bool,
        "cpu_description": str,
        "device_type": str,
        "device_unique_identifier": str,
        # format: uint64
        "external_free_storage": int,
        # format: uint64
        "external_storage_size": int,
        "family": str,
        # format: uint64
        "free_memory": int,
        # format: uint64
        "free_storage": int,
        "locale": str,
        "low_memory": bool,
        "manufacturer": str,
        # format: uint64
        "memory_size": int,
        "model": str,
        "model_id": str,
        "name": str,
        "online": bool,
        "orientation": str,
        # format: uint64
        "processor_count": int,
        # format: uint64
        "processor_frequency": int,
        # format: double
        "screen_density": float,
        # format: uint64
        "screen_dpi": int,
        # format: uint64
        "screen_height_pixels": int,
        "screen_resolution": str,
        # format: uint64
        "screen_width_pixels": int,
        "simulator": bool,
        # format: uint64
        "storage_size": int,
        "supports_accelerometer": bool,
        "supports_audio": bool,
        "supports_gyroscope": bool,
        "supports_location_service": bool,
        "supports_vibration": bool,
        "timezone": str,
        # format: uint64
        "usable_memory": int,
        # format: uuid
        "uuid": str,
    },
)

#  Wrapper around a UUID with slightly different formatting.
# format: uuid
EventId = str

#  An event processing error.
EventProcessingError = typing.TypedDict(
    "EventProcessingError",
    {
        "name": str,
        "type": str,
        "value": typing.Any,
    },
)

# The type of an event.
#
# The event type determines how Sentry handles the event and has an impact on processing, rate limiting, and quotas. There are three fundamental classes of event types:
#
# - **Error monitoring events** (`default`, `error`): Processed and grouped into unique issues based on their exception stack traces and error messages. - **Security events** (`csp`, `hpkp`, `expectct`, `expectstaple`): Derived from Browser security violation reports and grouped into unique issues based on the endpoint and violation. SDKs do not send such events. - **Transaction events** (`transaction`): Contain operation spans and collected into traces for performance monitoring.
EventType = str

#  A single exception.
#
#  Multiple values inside of an [event](#typedef-Event) represent chained exceptions and should be sorted oldest to newest. For example, consider this Python code snippet:
#
#  ```python
#  try:
#      raise Exception("random boring invariant was not met!")
#  except Exception as e:
#      raise ValueError("something went wrong, help!") from e
#  ```
#
#  `Exception` would be described first in the values list, followed by a description of `ValueError`:
#
#  ```json
#  {
#    "exception": {
#      "values": [
#        {"type": "Exception": "value": "random boring invariant was not met!"},
#        {"type": "ValueError", "value": "something went wrong, help!"},
#      ]
#    }
#  }
#  ```
Exception = typing.TypedDict(
    "Exception",
    {
        "mechanism": typing.Union["Mechanism", None],
        "module": str,
        "stacktrace": typing.Union["Stacktrace", None],
        "thread_id": typing.Union["ThreadId", None],
        "type": str,
        "value": typing.Union["JsonLenientString", None],
    },
)

#  A fingerprint value.
Fingerprint = typing.List[str]

#  Holds information about a single stacktrace frame.
#
#  Each object should contain **at least** a `filename`, `function` or `instruction_addr`
#  attribute. All values are optional, but recommended.
Frame = typing.TypedDict(
    "Frame",
    {
        "abs_path": typing.Union["NativeImagePath", None],
        "addr_mode": str,
        # format: uint64
        "colno": int,
        "context_line": str,
        "filename": typing.Union["NativeImagePath", None],
        "function": str,
        "function_id": typing.Union["Addr", None],
        "image_addr": typing.Union["Addr", None],
        "in_app": bool,
        "instruction_addr": typing.Union["Addr", None],
        # format: uint64
        "lineno": int,
        "lock": typing.Union["LockReason", None],
        "module": str,
        "package": str,
        "platform": str,
        "post_context": typing.List[str],
        "pre_context": typing.List[str],
        "raw_function": str,
        "stack_start": bool,
        "symbol": str,
        "symbol_addr": typing.Union["Addr", None],
        "vars": typing.Union["FrameVars", None],
    },
)

#  Frame local variables.
FrameVars = typing.Mapping[str, typing.Any]

#  Geographical location of the end user or device.
Geo = typing.TypedDict(
    "Geo",
    {
        "city": str,
        "country_code": str,
        "region": str,
        "subdivision": str,
    },
)

#  GPU information.
#
#  Example:
#
#  ```json
#  "gpu": {
#    "name": "AMD Radeon Pro 560",
#    "vendor_name": "Apple",
#    "memory_size": 4096,
#    "api_type": "Metal",
#    "multi_threaded_rendering": true,
#    "version": "Metal",
#    "npot_support": "Full"
#  }
#  ```
GpuContext = typing.TypedDict(
    "GpuContext",
    {
        "api_type": str,
        "graphics_shader_level": str,
        "id": typing.Any,
        # format: uint64
        "max_texture_size": int,
        # format: uint64
        "memory_size": int,
        "multi_threaded_rendering": bool,
        "name": str,
        "npot_support": str,
        "supports_compute_shaders": bool,
        "supports_draw_call_instancing": bool,
        "supports_geometry_shaders": bool,
        "supports_ray_tracing": bool,
        "vendor_id": str,
        "vendor_name": str,
        "version": str,
    },
)

#  A "into-string" type that normalizes header names.
HeaderName = str

#  A "into-string" type that normalizes header values.
HeaderValue = str

#  A map holding headers.
Headers = typing.Any

# Controls the mechanism by which the `instruction_addr` of a [`Stacktrace`] [`Frame`] is adjusted.
#
# The adjustment tries to transform *return addresses* to *call addresses* for symbolication. Typically, this adjustment needs to be done for all frames but the first, as the first frame is usually taken directly from the cpu context of a hardware exception or a suspended thread and the stack trace is created from that.
#
# When the stack walking implementation truncates frames from the top, `"all"` frames should be adjusted. In case the stack walking implementation already does the adjustment when producing stack frames, `"none"` should be used here.
InstructionAddrAdjustment = str

#  A "into-string" type of value. All non-string values are serialized as JSON.
JsonLenientString = str

#  A debug image consisting of source files for a JVM based language.
#
#  Examples:
#
#  ```json
#  {
#    "type": "jvm",
#    "debug_id": "395835f4-03e0-4436-80d3-136f0749a893"
#  }
#  ```
JvmDebugImage = typing.TypedDict(
    "JvmDebugImage",
    {
        "debug_id": typing.Union["DebugId", None],
    },
)

# Severity level of an event or breadcrumb.
Level = str

#  Represents an instance of a held lock (java monitor object) in a thread.
LockReason = typing.TypedDict(
    "LockReason",
    {
        "address": str,
        "class_name": str,
        "package_name": str,
        "thread_id": typing.Union["ThreadId", None],
        "type": typing.Union["LockReasonType", None],
    },
)

# Possible lock types responsible for a thread's blocked state
LockReasonType = str

#  A log entry message.
#
#  A log message is similar to the `message` attribute on the event itself but
#  can additionally hold optional parameters.
#
#  ```json
#  {
#    "message": {
#      "message": "My raw message with interpreted strings like %s",
#      "params": ["this"]
#    }
#  }
#  ```
#
#  ```json
#  {
#    "message": {
#      "message": "My raw message with interpreted strings like {foo}",
#      "params": {"foo": "this"}
#    }
#  }
#  ```
LogEntry = typing.TypedDict(
    "LogEntry",
    {
        "formatted": typing.Union["Message", None],
        "message": typing.Union["Message", None],
        "params": typing.Any,
    },
)

#  Mach exception information.
MachException = typing.TypedDict(
    "MachException",
    {
        # format: uint64
        "code": int,
        # format: int64
        "exception": int,
        "name": str,
        # format: uint64
        "subcode": int,
    },
)

#  The mechanism by which an exception was generated and handled.
#
#  The exception mechanism is an optional field residing in the [exception](#typedef-Exception).
#  It carries additional information about the way the exception was created on the target system.
#  This includes general exception values obtained from the operating system or runtime APIs, as
#  well as mechanism-specific values.
Mechanism = typing.TypedDict(
    "Mechanism",
    {
        "data": typing.Mapping[str, typing.Any],
        "description": str,
        # format: uint64
        "exception_id": int,
        "handled": bool,
        "help_link": str,
        "is_exception_group": bool,
        "meta": typing.Union["MechanismMeta", None],
        # format: uint64
        "parent_id": int,
        "source": str,
        "synthetic": bool,
        "type": str,
    },
)

#  Operating system or runtime meta information to an exception mechanism.
#
#  The mechanism metadata usually carries error codes reported by the runtime or operating system,
#  along with a platform-dependent interpretation of these codes. SDKs can safely omit code names
#  and descriptions for well-known error codes, as it will be filled out by Sentry. For
#  proprietary or vendor-specific error codes, adding these values will give additional
#  information to the user.
MechanismMeta = typing.TypedDict(
    "MechanismMeta",
    {
        "errno": typing.Union["CError", None],
        "mach_exception": typing.Union["MachException", None],
        "ns_error": typing.Union["NsError", None],
        "signal": typing.Union["PosixSignal", None],
    },
)

Message = str

#  Monitor information.
MonitorContext = typing.Mapping[str, typing.Any]

#  A generic (new-style) native platform debug information file.
#
#  The `type` key must be one of:
#
#  - `macho`
#  - `elf`: ELF images are used on Linux platforms. Their structure is identical to other native images.
#  - `pe`
#
#  Examples:
#
#  ```json
#  {
#    "type": "elf",
#    "code_id": "68220ae2c65d65c1b6aaa12fa6765a6ec2f5f434",
#    "code_file": "/lib/x86_64-linux-gnu/libgcc_s.so.1",
#    "debug_id": "e20a2268-5dc6-c165-b6aa-a12fa6765a6e",
#    "image_addr": "0x7f5140527000",
#    "image_size": 90112,
#    "image_vmaddr": "0x40000",
#    "arch": "x86_64"
#  }
#  ```
#
#  ```json
#  {
#    "type": "pe",
#    "code_id": "57898e12145000",
#    "code_file": "C:\\Windows\\System32\\dbghelp.dll",
#    "debug_id": "9c2a902b-6fdf-40ad-8308-588a41d572a0-1",
#    "debug_file": "dbghelp.pdb",
#    "image_addr": "0x70850000",
#    "image_size": "1331200",
#    "image_vmaddr": "0x40000",
#    "arch": "x86"
#  }
#  ```
#
#  ```json
#  {
#    "type": "macho",
#    "debug_id": "84a04d24-0e60-3810-a8c0-90a65e2df61a",
#    "debug_file": "libDiagnosticMessagesClient.dylib",
#    "code_file": "/usr/lib/libDiagnosticMessagesClient.dylib",
#    "image_addr": "0x7fffe668e000",
#    "image_size": 8192,
#    "image_vmaddr": "0x40000",
#    "arch": "x86_64",
#  }
#  ```
NativeDebugImage = typing.TypedDict(
    "NativeDebugImage",
    {
        "arch": str,
        "code_file": typing.Union["NativeImagePath", None],
        "code_id": typing.Union["CodeId", None],
        "debug_checksum": str,
        "debug_file": typing.Union["NativeImagePath", None],
        "debug_id": typing.Union["DebugId", None],
        "image_addr": typing.Union["Addr", None],
        # format: uint64
        "image_size": int,
        "image_vmaddr": typing.Union["Addr", None],
    },
)

#  A type for strings that are generally paths, might contain system user names, but still cannot
#  be stripped liberally because it would break processing for certain platforms.
#
#  Those strings get special treatment in our PII processor to avoid stripping the basename.
NativeImagePath = str

#  Contains NEL report information.
#
#  Network Error Logging (NEL) is a browser feature that allows reporting of failed network
#  requests from the client side. See the following resources for more information:
#
#  - [W3C Editor's Draft](https://w3c.github.io/network-error-logging/)
#  - [MDN](https://developer.mozilla.org/en-US/docs/Web/HTTP/Network_Error_Logging)
NelContext = typing.TypedDict(
    "NelContext",
    {
        # format: uint64
        "elapsed_time": int,
        "error_type": str,
        "phase": typing.Union["NetworkReportPhases", None],
        # format: double
        "sampling_fraction": float,
        "server_ip": typing.Union["String", None],
    },
)

#  Describes which phase the error occurred in.
NetworkReportPhases = typing.Union[
    typing.Mapping[str, typing.Any],
    typing.Mapping[str, typing.Any],
    typing.Mapping[str, typing.Any],
    str,
]

#  NSError informaiton.
NsError = typing.TypedDict(
    "NsError",
    {
        # format: int64
        "code": int,
        "domain": str,
    },
)

#  Operating system information.
#
#  OS context describes the operating system on which the event was created. In web contexts, this
#  is the operating system of the browser (generally pulled from the User-Agent string).
OsContext = typing.TypedDict(
    "OsContext",
    {
        "build": str,
        "kernel_version": str,
        "name": str,
        "raw_description": str,
        "rooted": bool,
        "version": str,
    },
)

#  OpenTelemetry Context
#
#  If an event has this context, it was generated from an OpenTelemetry signal (trace, metric, log).
OtelContext = typing.TypedDict(
    "OtelContext",
    {
        "attributes": typing.Mapping[str, typing.Any],
        "resource": typing.Mapping[str, typing.Any],
    },
)

#  POSIX signal with optional extended data.
#
#  On Apple systems, signals also carry a code in addition to the signal number describing the
#  signal in more detail. On Linux, this code does not exist.
PosixSignal = typing.TypedDict(
    "PosixSignal",
    {
        # format: int64
        "code": int,
        "code_name": str,
        "name": str,
        # format: int64
        "number": int,
    },
)

#  Profile context
ProfileContext = typing.TypedDict(
    "ProfileContext",
    {
        "profile_id": typing.Union["EventId", None],
    },
)

#  Proguard mapping file.
#
#  Proguard images refer to `mapping.txt` files generated when Proguard obfuscates function names. The Java SDK integrations assign this file a unique identifier, which has to be included in the list of images.
ProguardDebugImage = typing.TypedDict(
    "ProguardDebugImage",
    {
        # format: uuid
        "uuid": str,
    },
)

#  A stack trace of a single thread.
#
#  A stack trace contains a list of frames, each with various bits (most optional) describing the
#  context of that frame. Frames should be sorted from oldest to newest.
#
#  For the given example program written in Python:
#
#  ```python
#  def foo():
#      my_var = 'foo'
#      raise ValueError()
#
#  def main():
#      foo()
#  ```
#
#  A minimalistic stack trace for the above program in the correct order:
#
#  ```json
#  {
#    "frames": [
#      {"function": "main"},
#      {"function": "foo"}
#    ]
#  }
#  ```
#
#  The top frame fully symbolicated with five lines of source context:
#
#  ```json
#  {
#    "frames": [{
#      "in_app": true,
#      "function": "myfunction",
#      "abs_path": "/real/file/name.py",
#      "filename": "file/name.py",
#      "lineno": 3,
#      "vars": {
#        "my_var": "'value'"
#      },
#      "pre_context": [
#        "def foo():",
#        "  my_var = 'foo'",
#      ],
#      "context_line": "  raise ValueError()",
#      "post_context": [
#        "",
#        "def main():"
#      ],
#    }]
#  }
#  ```
#
#  A minimal native stack trace with register values. Note that the `package` event attribute must
#  be "native" for these frames to be symbolicated.
#
#  ```json
#  {
#    "frames": [
#      {"instruction_addr": "0x7fff5bf3456c"},
#      {"instruction_addr": "0x7fff5bf346c0"},
#    ],
#    "registers": {
#      "rip": "0x00007ff6eef54be2",
#      "rsp": "0x0000003b710cd9e0"
#    }
#  }
#  ```
RawStacktrace = typing.TypedDict(
    "RawStacktrace",
    {
        "frames": typing.List[typing.Union["Frame", None]],
        "instruction_addr_adjustment": typing.Union["InstructionAddrAdjustment", None],
        "lang": str,
        "registers": typing.Mapping[str, typing.Union["RegVal", None]],
        "snapshot": bool,
    },
)

RegVal = str

#  Replay context.
#
#  The replay context contains the replay_id of the session replay if the event
#  occurred during a replay. The replay_id is added onto the dynamic sampling context
#  on the javascript SDK which propagates it through the trace. In relay, we take
#  this value from the DSC and create a context which contains only the replay_id
#  This context is never set on the client for events, only on relay.
ReplayContext = typing.TypedDict(
    "ReplayContext",
    {
        "replay_id": typing.Union["EventId", None],
    },
)

#  Http request information.
#
#  The Request interface contains information on a HTTP request related to the event. In client
#  SDKs, this can be an outgoing request, or the request that rendered the current web page. On
#  server SDKs, this could be the incoming web request that is being handled.
#
#  The data variable should only contain the request body (not the query string). It can either be
#  a dictionary (for standard HTTP requests) or a raw request body.
#
#  ### Ordered Maps
#
#  In the Request interface, several attributes can either be declared as string, object, or list
#  of tuples. Sentry attempts to parse structured information from the string representation in
#  such cases.
#
#  Sometimes, keys can be declared multiple times, or the order of elements matters. In such
#  cases, use the tuple representation over a plain object.
#
#  Example of request headers as object:
#
#  ```json
#  {
#    "content-type": "application/json",
#    "accept": "application/json, application/xml"
#  }
#  ```
#
#  Example of the same headers as list of tuples:
#
#  ```json
#  [
#    ["content-type", "application/json"],
#    ["accept", "application/json"],
#    ["accept", "application/xml"]
#  ]
#  ```
#
#  Example of a fully populated request object:
#
#  ```json
#  {
#    "request": {
#      "method": "POST",
#      "url": "http://absolute.uri/foo",
#      "query_string": "query=foobar&page=2",
#      "data": {
#        "foo": "bar"
#      },
#      "cookies": "PHPSESSID=298zf09hf012fh2; csrftoken=u32t4o3tb3gg43; _gat=1;",
#      "headers": {
#        "content-type": "text/html"
#      },
#      "env": {
#        "REMOTE_ADDR": "192.168.0.1"
#      }
#    }
#  }
#  ```
Request = typing.TypedDict(
    "Request",
    {
        "api_target": str,
        # format: uint64
        "body_size": int,
        "cookies": typing.Union["Cookies", None],
        "data": typing.Any,
        "env": typing.Mapping[str, typing.Any],
        "fragment": str,
        "headers": typing.Union["Headers", None],
        "inferred_content_type": str,
        "method": str,
        "protocol": str,
        "query_string": typing.Union[
            typing.Union[
                str,
                typing.Union[typing.Mapping[str, typing.Any], typing.List[typing.Tuple[str, str]]],
            ],
            None,
        ],
        "url": str,
    },
)

#  Response interface that contains information on a HTTP response related to the event.
#
#  The data variable should only contain the response body. It can either be
#  a dictionary (for standard HTTP responses) or a raw response body.
ResponseContext = typing.TypedDict(
    "ResponseContext",
    {
        # format: uint64
        "body_size": int,
        "cookies": typing.Union["Cookies", None],
        "data": typing.Any,
        "headers": typing.Union["Headers", None],
        "inferred_content_type": str,
        # format: uint64
        "status_code": int,
    },
)

#  The route in the application, set by React Native SDK.
Route = typing.TypedDict(
    "Route",
    {
        "name": str,
        "params": typing.Mapping[str, typing.Any],
    },
)

#  Runtime information.
#
#  Runtime context describes a runtime in more detail. Typically, this context is present in
#  `contexts` multiple times if multiple runtimes are involved (for instance, if you have a
#  JavaScript application running on top of JVM).
RuntimeContext = typing.TypedDict(
    "RuntimeContext",
    {
        "build": str,
        "name": str,
        "raw_description": str,
        "version": str,
    },
)

#  A debug image pointing to a source map.
#
#  Examples:
#
#  ```json
#  {
#    "type": "sourcemap",
#    "code_file": "https://example.com/static/js/main.min.js",
#    "debug_id": "395835f4-03e0-4436-80d3-136f0749a893"
#  }
#  ```
#
#  **Note:** Stack frames and the correlating entries in the debug image here
#  for `code_file`/`abs_path` are not PII stripped as they need to line up
#  perfectly for source map processing.
SourceMapDebugImage = typing.TypedDict(
    "SourceMapDebugImage",
    {
        "code_file": str,
        "debug_file": str,
        "debug_id": typing.Union["DebugId", None],
    },
)

#  A 16-character hex string as described in the W3C trace context spec.
SpanId = str

# Trace status.
#
# Values from <https://github.com/open-telemetry/opentelemetry-specification/blob/8fb6c14e4709e75a9aaa64b0dbbdf02a6067682a/specification/api-tracing.md#status> Mapping to HTTP from <https://github.com/open-telemetry/opentelemetry-specification/blob/8fb6c14e4709e75a9aaa64b0dbbdf02a6067682a/specification/data-http.md#status>
SpanStatus = str

Stacktrace = typing.Any

String = str

#  Holds information about the system SDK.
#
#  This is relevant for iOS and other platforms that have a system
#  SDK.  Not to be confused with the client SDK.
SystemSdkInfo = typing.TypedDict(
    "SystemSdkInfo",
    {
        "sdk_name": str,
        # format: uint64
        "version_major": int,
        # format: uint64
        "version_minor": int,
        # format: uint64
        "version_patchlevel": int,
    },
)

TagEntry = typing.Tuple[str, str]

#  Manual key/value tag pairs.
Tags = typing.Any

#  A process thread of an event.
#
#  The Threads Interface specifies threads that were running at the time an event happened. These threads can also contain stack traces.
#
#  An event may contain one or more threads in an attribute named `threads`.
#
#  The following example illustrates the threads part of the event payload and omits other attributes for simplicity.
#
#  ```json
#  {
#    "threads": {
#      "values": [
#        {
#          "id": "0",
#          "name": "main",
#          "crashed": true,
#          "stacktrace": {}
#        }
#      ]
#    }
#  }
#  ```
Thread = typing.TypedDict(
    "Thread",
    {
        "crashed": bool,
        "current": bool,
        "held_locks": typing.Mapping[str, typing.Union["LockReason", None]],
        "id": typing.Union["ThreadId", None],
        "main": bool,
        "name": str,
        "stacktrace": typing.Union["Stacktrace", None],
        "state": str,
    },
)

#  Represents a thread id.
# format: uint64
ThreadId = typing.Union[int, str]

# Can be a ISO-8601 formatted string or a unix timestamp in seconds (floating point values allowed).
#
# Must be UTC.
# format: double
Timestamp = typing.Union[float, str]

#  Trace context
TraceContext = typing.TypedDict(
    "TraceContext",
    {
        # format: double
        "client_sample_rate": float,
        "data": typing.Union["Data", None],
        # format: double
        "exclusive_time": float,
        "op": str,
        "origin": str,
        "parent_span_id": typing.Union["SpanId", None],
        "sampled": bool,
        "span_id": typing.Union["SpanId", None],
        "status": typing.Union["SpanStatus", None],
        "trace_id": typing.Union["TraceId", None],
    },
)

#  A 32-character hex string as described in the W3C trace context spec.
TraceId = str

#  Additional information about the name of the transaction.
TransactionInfo = typing.TypedDict(
    "TransactionInfo",
    {
        "changes": typing.List[typing.Union["TransactionNameChange", None]],
        "original": str,
        # format: uint64
        "propagations": int,
        "source": typing.Union["TransactionSource", None],
    },
)

TransactionNameChange = typing.TypedDict(
    "TransactionNameChange",
    {
        # format: uint64
        "propagations": int,
        "source": typing.Union["TransactionSource", None],
        "timestamp": typing.Union["Timestamp", None],
    },
)

# Describes how the name of the transaction was determined.
TransactionSource = str

#  Information about the user who triggered an event.
#
#  ```json
#  {
#    "user": {
#      "id": "unique_id",
#      "username": "my_user",
#      "email": "foo@example.com",
#      "ip_address": "127.0.0.1",
#      "subscription": "basic"
#    }
#  }
#  ```
User = typing.TypedDict(
    "User",
    {
        "data": typing.Mapping[str, typing.Any],
        "email": str,
        "geo": typing.Union["Geo", None],
        "id": str,
        "ip_address": typing.Union["String", None],
        "name": str,
        "segment": str,
        "sentry_user": str,
        "username": str,
    },
)

#  Feedback context.
#
#  This contexts contains user feedback specific attributes.
#  We don't PII scrub contact_email as that is provided by the user.
#  TODO(jferg): rename to FeedbackContext once old UserReport logic is deprecated.
UserReportV2Context = typing.TypedDict(
    "UserReportV2Context",
    {
        "contact_email": str,
        "message": str,
    },
)
