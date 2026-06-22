from pydantic import BaseModel, Field
from typing import Optional


class DocumentId(BaseModel):
    number: Optional[str] = Field(None, description="Número do documento/contrato")
    ig: Optional[str] = Field(None, description="Número IG do documento")


class Party(BaseModel):
    name: Optional[str] = Field(None, description="Nome da empresa/orgão")
    cpf_cnpj: Optional[str] = Field(None, description="CPF ou CNPJ da empresa/orgão")


class Signatories(BaseModel):
    contracting_party: Optional[str] = Field(None, description="Nome do signatário contratante")
    contracted_party: Optional[str] = Field(None, description="Nome do signatário contratado")


class Signature(BaseModel):
    date: Optional[str] = Field(None, description="Data da assinatura no formato DD/MM/YYYY")
    signatories: Signatories = Field(default_factory=Signatories)


class LegalUnit(BaseModel):
    name: Optional[str] = Field(None, description="Nome do responsável pela unidade jurídica")
    role: Optional[str] = Field(None, description="Cargo/função do responsável jurídico")


class ContractExtract(BaseModel):
    document_id: DocumentId = Field(default_factory=DocumentId)
    contracting_party: Party = Field(default_factory=Party, description="Pessoa jurídica contratante")
    contracted_party: Party = Field(default_factory=Party, description="Pessoa jurídica contratada")
    validity: Optional[str] = Field(None, description="Prazo de vigência do contrato")
    global_value: Optional[str] = Field(None, description="Valor global do contrato")
    signature: Signature = Field(default_factory=Signature)
    legal_unit: LegalUnit = Field(default_factory=LegalUnit)

    @classmethod
    def json_schema_str(cls) -> str:
        """Returns the JSON schema as a formatted string for prompt injection."""
        import json
        return json.dumps(cls.model_json_schema(), ensure_ascii=False, indent=2)

    @classmethod
    def empty_example(cls) -> str:
        """Returns an empty JSON structure for prompt injection."""
        import json
        template = {
            "document_id": {"number": None, "ig": None},
            "contracting_party": {"name": None, "cpf_cnpj": None},
            "contracted_party": {"name": None, "cpf_cnpj": None},
            "validity": None,
            "global_value": None,
            "signature": {
                "date": None,
                "signatories": {
                    "contracting_party": None,
                    "contracted_party": None,
                },
            },
            "legal_unit": {"name": None, "role": None},
        }
        return json.dumps(template, ensure_ascii=False, indent=2)