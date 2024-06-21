import contextlib
import keyword
import sys
from typing import Iterable, Sequence

from google.protobuf.compiler import plugin_pb2 as plugin_pb2

# TODO: Make this compatible with tools
__version__ = "0.1.0"

from google.protobuf.descriptor_pb2 import (
    DescriptorProto,
    EnumDescriptorProto,
    FieldDescriptorProto,
    FileDescriptorProto,
)
from mypy_protobuf.main import Descriptors, PkgWriter, SourceCodeLocation, is_scalar


class AdaptorPkgWrite(PkgWriter):
    def write_enums(
        self,
        enums: Iterable[EnumDescriptorProto],
        prefix: str,
        scl_prefix: SourceCodeLocation,
    ) -> None:
        raise NotImplementedError

    def write_extensions(
        self,
        extensions: Sequence[FieldDescriptorProto],
        scl_prefix: SourceCodeLocation,
    ) -> None:
        raise NotImplementedError

    def write_messages(
        self,
        messages: Iterable[DescriptorProto],
        prefix: str,
        scl_prefix: SourceCodeLocation,
    ) -> None:
        self._write_adaptors(messages, prefix, scl_prefix, "from")
        self._write_adaptors(messages, prefix, scl_prefix, "to")

    def _write_adaptors(
        self,
        messages: Iterable[DescriptorProto],
        prefix: str,
        scl_prefix: SourceCodeLocation,
        adaptor_kind: str,
    ) -> None:
        wl = self._write_line

        for i, desc in enumerate(messages):
            qualified_name = prefix + desc.name

            class_name = desc.name if not keyword.iskeyword(desc.name) else "_r_" + desc.name
            # HACK: control the local-ness by the grpc attribute for now.
            self.grpc = True
            msg_type = self._import_message(qualified_name)
            self.grpc = False

            scl = scl_prefix + [i]
            if adaptor_kind == "from":
                self._write_from_adaptor(class_name, desc, scl, msg_type, qualified_name)
            elif adaptor_kind == "to":
                self._write_to_adaptor(class_name, desc, scl, msg_type, qualified_name)
            wl("")

    def _write_from_adaptor(
        self,
        class_name: str,
        desc: DescriptorProto,
        scl: list[int],
        msg_type: str,
        qualified_name: str,
    ):
        wl = self._write_line

        class_name = f"From{class_name}Adaptor"
        wl(f"class {class_name}:")
        with self._indent():
            # Nested message adaptors
            self._write_adaptors(
                desc.nested_type,
                qualified_name + ".",
                scl + [DescriptorProto.NESTED_TYPE_FIELD_NUMBER],
                "from",
            )

            for field in desc.field:
                field_type = self.python_type(field)
                wl(f"def adapt_from_{field.name}(self, value: {field_type}): pass")

            wl("")
            wl(f"def adapt_from(self, proto: {msg_type}):")
            with self._indent():
                for field in desc.field:
                    with contextlib.ExitStack() as stack:
                        if field.label == FieldDescriptorProto.LABEL_REPEATED:
                            has_presence = False
                        elif self.fd.syntax == "proto3":
                            has_presence = field.HasField("oneof_index") or not is_scalar(field)
                        else:
                            has_presence = True

                        if has_presence:
                            wl(f"if proto.HasField({repr(field.name)}):")
                            stack.enter_context(self._indent())
                        if keyword.iskeyword(field.name):
                            wl(
                                f"self.adapt_from_{field.name}(getattr(proto, {repr(field.name)})) # noqa"
                            )
                        else:
                            wl(f"self.adapt_from_{field.name}(proto.{field.name})")
            wl("")

    def _write_to_adaptor(
        self,
        class_name: str,
        desc: DescriptorProto,
        scl: list[int],
        msg_type: str,
        qualified_name: str,
    ):
        wl = self._write_line

        class_name = f"To{class_name}Adaptor"
        wl(f"class {class_name}:")
        with self._indent():
            # Nested message adaptors
            self._write_adaptors(
                desc.nested_type,
                qualified_name + ".",
                scl + [DescriptorProto.NESTED_TYPE_FIELD_NUMBER],
                "to",
            )

            for field in desc.field:
                generic_field_type = self.python_type(field, generic_container=True)
                wl(
                    f"def apply_to_{field.name}(self, proto: {msg_type}, val: {self._import('typing', 'Optional')}[{generic_field_type}] = None):"
                )
                with self._indent():
                    wl(f"if val is not None:")
                    with self._indent():
                        proto_field = f"proto.{field.name}"
                        if keyword.iskeyword(field.name):
                            proto_field = "getattr(proto, {repr(field.name)})"

                        if field.type in {
                            FieldDescriptorProto.TYPE_MESSAGE,
                            FieldDescriptorProto.TYPE_GROUP,
                        }:
                            msg = self.descriptors.messages[field.type_name]
                            if msg.options.map_entry:
                                wl(f"{proto_field}.clear()")
                                wl(f"for k, v in val.items():")
                                with self._indent():
                                    if is_scalar(msg.field[1]):
                                        wl(f"{proto_field}[k] = v")
                                    else:
                                        wl(f"{proto_field}.get_or_create(k).MergeFrom(v)")
                        elif field.label == FieldDescriptorProto.LABEL_REPEATED:
                            wl(f"{proto_field}.clear()")
                            wl(f"{proto_field}.extend(val)")
                        else:
                            wl(f"{proto_field} = val")
                    wl("else:")
                    with self._indent():
                        wl(f"proto.ClearField({repr(field.name)})")
                wl("")

            wl("")
            wl(f"def apply_to(self, proto: {msg_type}):")
            with self._indent():
                for field in desc.field:
                    wl(f"self.apply_to_{field.name}(proto)")
            wl("")


# Largely borrowed from mypy_proto.
# TODO: Extract into a PR upstream if we like this approach.
def generate_adaptors(
    descriptors: Descriptors,
    response: plugin_pb2.CodeGeneratorResponse,
    quiet: bool,
    readable_stubs: bool,
    relax_strict_optional_primitives: bool,
):
    for name, fd in descriptors.to_generate.items():
        pkg_writer = AdaptorPkgWrite(
            fd,
            descriptors,
            readable_stubs,
            relax_strict_optional_primitives,
            grpc=False,
        )

        pkg_writer.write_messages(
            fd.message_type, "." + fd.package + ".", [FileDescriptorProto.MESSAGE_TYPE_FIELD_NUMBER]
        )

        assert name == fd.name
        assert fd.name.endswith(".proto")
        output = response.file.add()
        output.name = fd.name[:-6].replace("-", "_").replace(".", "/") + "_adaptors.py"
        output.content = pkg_writer.write()
        if not quiet:
            print("Writing adaptors to", output.name, file=sys.stderr)


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-V", "--version"):
        print("protobuf-adaptors " + __version__)
        sys.exit(0)

    data = sys.stdin.buffer.read()
    request = plugin_pb2.CodeGeneratorRequest()
    request.ParseFromString(data)
    response = plugin_pb2.CodeGeneratorResponse()

    d = Descriptors(request)
    generate_adaptors(
        d,
        response,
        "quiet" in request.parameter,
        "readable_stubs" in request.parameter,
        "relax_strict_optional_primitives" in request.parameter,
    )

    output = response.SerializeToString()
    sys.stdout.buffer.write(output)


if __name__ == "__main__":
    main()
