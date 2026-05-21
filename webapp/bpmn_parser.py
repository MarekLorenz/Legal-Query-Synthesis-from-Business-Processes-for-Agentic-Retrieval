from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any


TASK_TAGS = {
    "task",
    "userTask",
    "serviceTask",
    "manualTask",
    "businessRuleTask",
    "scriptTask",
    "sendTask",
    "receiveTask",
}


def local_name(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def element_description(element: ET.Element) -> str:
    parts: list[str] = []
    for child in element:
        child_name = local_name(child.tag)
        if child_name == "documentation" and child.text and child.text.strip():
            parts.append(child.text.strip())

    name = (element.get("name") or "").strip()
    if parts:
        return " ".join(parts)
    return name or (element.get("id") or "Unnamed element")


def direct_children(parent: ET.Element, tag_names: set[str]) -> list[ET.Element]:
    return [child for child in list(parent) if local_name(child.tag) in tag_names]


def normalize_name(value: str) -> str:
    value = re.sub(r"^\s*\d+(?:\.\d+)?\s*[\.\-:]?\s*", "", str(value))
    return re.sub(r"\s+", " ", value).strip().lower()


@dataclass
class BpmnTaskIndex:
    by_id: dict[str, ET.Element]
    by_number: dict[int, ET.Element]
    by_name: dict[str, ET.Element]


def index_bpmn_tasks(process_element: ET.Element) -> BpmnTaskIndex:
    by_id: dict[str, ET.Element] = {}
    by_number: dict[int, ET.Element] = {}
    by_name: dict[str, ET.Element] = {}

    for element in process_element.iter():
        if local_name(element.tag) not in TASK_TAGS:
            continue
        element_id = element.get("id")
        if not element_id:
            continue
        by_id[element_id] = element
        name = (element.get("name") or "").strip()
        if name:
            by_name[normalize_name(name)] = element
            match = re.match(r"^(\d+)\.", name)
            if match:
                by_number[int(match.group(1))] = element

    return BpmnTaskIndex(by_id=by_id, by_number=by_number, by_name=by_name)


def low_node_id(item: dict[str, Any]) -> str:
    number = str(item["number"]).replace(".", "_")
    return f"Low_{number}"


def medium_node_id(item: dict[str, Any], bpmn_tasks: BpmnTaskIndex | None = None) -> str:
    number = int(item["number"])
    if bpmn_tasks is not None and number in bpmn_tasks.by_number:
        element = bpmn_tasks.by_number[number]
        return element.get("id") or f"Task_{number}"

    normalized = normalize_name(item.get("name", ""))
    if bpmn_tasks is not None and normalized in bpmn_tasks.by_name:
        element = bpmn_tasks.by_name[normalized]
        return element.get("id") or f"Task_{number}"

    slug = re.sub(r"[^a-zA-Z0-9]+", "", str(item["name"]))
    return f"Task_{slug}"


def build_process_model_from_json(raw: dict[str, Any], bpmn_tasks: BpmnTaskIndex | None = None) -> dict[str, Any]:
    subprocesses: list[dict[str, Any]] = []
    for medium in raw.get("medium_levels", []):
        children = [
            {
                "id": low_node_id(low),
                "number": str(low["number"]),
                "name": low["name"],
                "description": low["description"],
                "level": "task",
                "parentNumber": str(medium["number"]),
                "diagramElement": False,
            }
            for low in raw.get("low_levels", [])
            if int(low["category_number"]) == int(medium["number"])
        ]
        subprocesses.append(
            {
                "id": medium_node_id(medium, bpmn_tasks),
                "number": str(medium["number"]),
                "name": medium["name"],
                "description": medium["description"],
                "level": "subprocess",
                "diagramElement": True,
                "children": children,
            }
        )

    top_level = raw.get("top_level") or {}
    return {
        "topLevel": {
            "id": "ProcessRoot",
            "name": top_level.get("name", "Process"),
            "description": top_level.get("description", ""),
            "level": "process",
            "diagramElement": False,
        },
        "subprocesses": subprocesses,
    }


def build_process_model_from_bpmn(process_element: ET.Element, bpmn_tasks: BpmnTaskIndex) -> dict[str, Any]:
    process_id = process_element.get("id") or "ProcessRoot"
    process_name = process_element.get("name") or "Process"
    process_description = element_description(process_element)

    subprocesses: list[dict[str, Any]] = []
    nested_subprocesses = direct_children(process_element, {"subProcess"})

    if nested_subprocesses:
        for subprocess_index, subprocess_element in enumerate(nested_subprocesses, start=1):
            children = []
            for task_index, task_element in enumerate(direct_children(subprocess_element, TASK_TAGS), start=1):
                children.append(
                    {
                        "id": task_element.get("id") or f"{subprocess_element.get('id')}_Task_{task_index}",
                        "number": f"{subprocess_index}.{task_index}",
                        "name": task_element.get("name") or f"Task {task_index}",
                        "description": element_description(task_element),
                        "level": "task",
                        "parentNumber": str(subprocess_index),
                        "diagramElement": True,
                    }
                )
            subprocesses.append(
                {
                    "id": subprocess_element.get("id") or f"SubProcess_{subprocess_index}",
                    "number": str(subprocess_index),
                    "name": subprocess_element.get("name") or f"Subprocess {subprocess_index}",
                    "description": element_description(subprocess_element),
                    "level": "subprocess",
                    "diagramElement": True,
                    "children": children,
                }
            )
    else:
        ordered_tasks = sorted(
            bpmn_tasks.by_id.values(),
            key=lambda element: (
                int(re.match(r"^(\d+)\.", (element.get("name") or "")).group(1))
                if re.match(r"^(\d+)\.", (element.get("name") or ""))
                else 10_000
            ),
        )
        for task_element in ordered_tasks:
            name = (task_element.get("name") or "").strip()
            match = re.match(r"^(\d+)\.\s*(.+)$", name)
            number = match.group(1) if match else str(len(subprocesses) + 1)
            display_name = match.group(2).strip() if match else name or task_element.get("id")
            subprocesses.append(
                {
                    "id": task_element.get("id") or f"Task_{number}",
                    "number": str(number),
                    "name": display_name,
                    "description": element_description(task_element),
                    "level": "subprocess",
                    "diagramElement": True,
                    "children": [],
                }
            )

    return {
        "topLevel": {
            "id": process_id,
            "name": process_name,
            "description": process_description,
            "level": "process",
            "diagramElement": False,
        },
        "subprocesses": subprocesses,
    }


def build_process_model(xml_content: str, process_json: dict[str, Any] | None = None) -> dict[str, Any]:
    root = ET.fromstring(xml_content)
    process_element = next((el for el in root.iter() if local_name(el.tag) == "process"), None)
    if process_element is None:
        raise ValueError("No BPMN process element found in the uploaded XML.")

    bpmn_tasks = index_bpmn_tasks(process_element)
    if process_json:
        model = build_process_model_from_json(process_json, bpmn_tasks)
        model["topLevel"]["id"] = process_element.get("id") or model["topLevel"]["id"]
        model["topLevel"]["name"] = process_json.get("top_level", {}).get("name") or model["topLevel"]["name"]
        model["topLevel"]["description"] = process_json.get("top_level", {}).get("description") or model["topLevel"]["description"]
        return model

    return build_process_model_from_bpmn(process_element, bpmn_tasks)


def parse_bpmn_process_model(xml_content: str) -> dict[str, Any]:
    return build_process_model(xml_content, process_json=None)


def load_optional_process_json(content: bytes) -> dict[str, Any] | None:
    if not content:
        return None
    try:
        return json.loads(content.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Process JSON is invalid: {exc}") from exc
