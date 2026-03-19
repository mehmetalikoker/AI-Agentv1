import pytest
from unittest.mock import MagicMock, patch
from logic import process_pdf


def test_process_pdf_creates_tool(mocker):
    mock_uploaded_file = MagicMock()
    mock_uploaded_file.getbuffer.return_value = b"fake pdf content"

    # Document Mock
    mock_doc = MagicMock()
    mock_doc.page_content = "Test content"
    mock_doc.metadata = {"source": "temp.pdf"}

    mocker.patch("logic.PyPDFLoader.load", return_value=[mock_doc])

    # Embedding and Vector Store mock
    mocker.patch("logic.OpenAIEmbeddings")
    mocker.patch("logic.FAISS.from_documents")

    # Run Function
    tool = process_pdf(mock_uploaded_file)

    # Validation
    assert tool.name == "pdf_search"
    print("Test were completed successfully.!")