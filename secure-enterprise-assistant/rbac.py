"""
rbac.py — Role-based access control (RBAC) for documents.

THE RULE (two checks, both must pass):
  1. CLEARANCE:  the document's classification must be <= the role's clearance.
  2. DEPARTMENT: the document must belong to "all" or to the user's own
                 department. (Admins skip this check.)

  Role        Clearance       Example
  ----------  --------------  -----------------------------------------
  guest       PUBLIC          company FAQ
  employee    INTERNAL        handbook + own department's docs
  manager     CONFIDENTIAL    own department's confidential reports
  admin       RESTRICTED      everything

LINE BY LINE:
  ROLE_CLEARANCE.get(session.role, Clearance.PUBLIC)
      An unknown role (typo, corrupted token, new role nobody mapped) gets the
      LOWEST clearance. Fail closed again.

  if doc.classification > clearance: return False
      Possible because Clearance is an IntEnum, so `>` just works.

  if session.role == "admin": return True
      Admins see every department. This runs AFTER the clearance check, so
      the order is deliberate and cheap checks come first.

  doc.department in ("all", session.department)
      A "confidential / finance" document is visible to managers in finance
      only. A manager in engineering has the clearance but not the department.

  filter_documents()
      Returns only the documents the session may read. The assistant calls
      this BEFORE retrieval. The retriever never even sees the other
      documents, so a clever prompt cannot make the LLM quote them.
      Filtering AFTER retrieval ("get the top 3, then hide the forbidden
      ones") is weaker: forbidden chunks can crowd out allowed ones, and the
      score of a hidden chunk can leak information.
"""

from typing import List

from models import Clearance, Document, Session

ROLE_CLEARANCE = {
    "guest":    Clearance.PUBLIC,
    "employee": Clearance.INTERNAL,
    "manager":  Clearance.CONFIDENTIAL,
    "admin":    Clearance.RESTRICTED,
}


def can_access(session: Session, doc: Document) -> bool:
    clearance = ROLE_CLEARANCE.get(session.role, Clearance.PUBLIC)

    if doc.classification > clearance:
        return False

    if session.role == "admin":
        return True

    return doc.department in ("all", session.department)


def filter_documents(session: Session, documents: List[Document]) -> List[Document]:
    return [d for d in documents if can_access(session, d)]
