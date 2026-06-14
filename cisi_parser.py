import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class CISIDoc:
    doc_id: int
    title: str = ""
    author: str = ""
    abstract: str = ""

    @property
    def text(self) -> str:
        return f"{self.title} {self.abstract}".strip()


@dataclass
class CISIQuery:
    query_id: int
    text: str = ""


def parse_cisi_all(content: str) -> Dict[int, CISIDoc]:
    docs: Dict[int, CISIDoc] = {}
    current: CISIDoc | None = None
    current_field = None

    for line in content.splitlines():
        if line.startswith(".I"):
            if current:
                docs[current.doc_id] = current
            current = CISIDoc(doc_id=int(line.split()[1]))
            current_field = None
        elif line.startswith(".T"):
            current_field = "T"
        elif line.startswith(".A"):
            current_field = "A"
        elif line.startswith(".W"):
            current_field = "W"
        elif line.startswith(".X"):
            current_field = None
        elif current and current_field:
            if current_field == "T":
                current.title = (current.title + " " + line).strip()
            elif current_field == "A":
                current.author = (current.author + " " + line).strip()
            elif current_field == "W":
                current.abstract = (current.abstract + " " + line).strip()

    if current:
        docs[current.doc_id] = current
    return docs


def parse_cisi_qry(content: str) -> Dict[int, CISIQuery]:
    queries: Dict[int, CISIQuery] = {}
    current: CISIQuery | None = None
    current_field = None

    for line in content.splitlines():
        if line.startswith(".I"):
            if current:
                queries[current.query_id] = current
            current = CISIQuery(query_id=int(line.split()[1]))
            current_field = None
        elif line.startswith(".W"):
            current_field = "W"
        elif line.startswith("."):
            current_field = None
        elif current and current_field == "W":
            current.text = (current.text + " " + line).strip()

    if current:
        queries[current.query_id] = current
    return queries


def parse_cisi_rel(content: str) -> Dict[int, List[int]]:
    rel: Dict[int, List[int]] = {}
    for line in content.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            qid, did = int(parts[0]), int(parts[1])
            rel.setdefault(qid, []).append(did)
    return rel
