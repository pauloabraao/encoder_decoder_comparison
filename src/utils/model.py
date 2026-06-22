from pydantic import BaseModel, ConfigDict, Field
from typing import Optional

class DocumentID(BaseModel):
    model_config = ConfigDict(extra="forbid")
    number: Optional[str]
    ig: Optional[str]


class Party(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Optional[str]
    cpf_cnpj: Optional[str]


class Signatories(BaseModel):
    model_config = ConfigDict(extra="forbid")
    contracting_party: Optional[str]
    contracted_party: Optional[str]


class Signature(BaseModel):
    model_config = ConfigDict(extra="forbid")
    date: Optional[str]
    signatories: Signatories


class LegalUnit(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Optional[str]
    role: Optional[str]


class ExtratoContrato(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_type: Optional[str]
    document_id: DocumentID
    contracting_party: Party
    contracted_party: Party
    validity: Optional[str]
    global_value: Optional[str]
    signature: Signature
    legal_unit: LegalUnit
