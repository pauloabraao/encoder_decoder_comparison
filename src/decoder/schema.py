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