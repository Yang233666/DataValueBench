"""Reusable DataValueBench benchmark implementation."""
import hashlib
import re
import unicodedata


def normalized_string(value):
    if value is None or value == []:
        return ""
    if not isinstance(value,str):
        raise ValueError("OpenML included fields must be strings or absent")
    return re.sub(r"\s+"," ",unicodedata.normalize("NFKC",value)).strip()


def collection(value):
    if value is None or value == []:return []
    if isinstance(value,dict):return [value]
    if isinstance(value,list) and all(isinstance(x,dict) for x in value):return value
    raise ValueError("invalid OpenML repeated element")


def canonical_workflow_text(response):
    root=response["flow"] if "flow" in response else response
    def visit(node,path,ancestor_objects,ancestor_flow_ids):
        lines=[]
        identity=node.get("id")
        if id(node) in ancestor_objects or (identity is not None and identity in ancestor_flow_ids):
            raise ValueError("BLOCKED: OpenML flow component cycle violates canonical serializer")
        objects=ancestor_objects|{id(node)};flow_ids=ancestor_flow_ids|({identity} if identity is not None else set())
        prefix="FLOW" if path=="ROOT" else "COMPONENT"
        if path!="ROOT":lines.append(f"COMPONENT_PATH: {path}")
        for label,key in (("CLASS","class_name"),("NAME","name"),("VERSION","external_version")):
            lines.append(f"{prefix}_{label}: {normalized_string(node.get(key))}")
        parameters=[]
        for parameter in collection(node.get("parameter")):
            name=normalized_string(parameter.get("name"))
            if not name:raise ValueError("parameter name absent")
            value=parameter.get("default_value")
            normalized="<MISSING>" if value is None or value==[] else normalized_string(value)
            parameters.append((name,normalized))
        if len({x[0] for x in parameters})!=len(parameters):raise ValueError("duplicate normalized parameter names")
        for name,value in sorted(parameters):lines.append(f"PARAM: {path}::{name}={value}")
        components=[]
        for component in collection(node.get("component")):
            identifier=normalized_string(component.get("identifier"))
            if not identifier or not isinstance(component.get("flow"),dict):raise ValueError("incomplete component identity/flow")
            child_lines=visit(component["flow"],f"{path}/{identifier}",objects,flow_ids)
            # Equal identifiers share the same path. Their complete emitted
            # subtree is therefore a permitted-field-only canonical secondary key.
            components.append((identifier,"\n".join(child_lines),child_lines))
        for identifier,subtree,child_lines in sorted(components,key=lambda x:(x[0],x[1])):
            lines.extend(child_lines)
        return lines
    return "\n".join(visit(root,"ROOT",set(),set()))


def text_sha256(text):return hashlib.sha256(text.encode("utf-8")).hexdigest()
