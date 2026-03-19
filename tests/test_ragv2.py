import pytest
import os
from unittest.mock import MagicMock, patch
from rag.agentwithragv2 import process_multiple_files
import logic
from unittest.mock import MagicMock
import pytest
from unittest.mock import MagicMock
import rag.agentwithragv2 as agentwithragv2


class TestXFrameworkAgent:

    @pytest.fixture
    def mock_uploaded_file(self):
        mock_file = MagicMock()
        mock_file.name = "test_doc.txt"
        mock_file.getbuffer.return_value = b"Test content"
        return mock_file

    def test_process_multiple_files_logic(self, mocker):
        mock_file = MagicMock()
        mock_file.name = "test_doc.txt"
        mock_file.getbuffer.return_value = b"test icerigi"

        mock_loader_instance = MagicMock()
        mock_loader_instance.load.return_value = [
            MagicMock(page_content="test verisi", metadata={"source": "test_doc.txt"})
        ]
        mock_loader_class = MagicMock(return_value=mock_loader_instance)

        agentwithragv2.UnstructuredFileLoader = mock_loader_class

        mocker.patch("os.makedirs")
        mocker.patch("os.path.exists", return_value=True)
        mocker.patch("os.path.isfile", return_value=True)
        mocker.patch("builtins.open", mocker.mock_open())
        mocker.patch("agentwithragv2.OpenAIEmbeddings")
        mocker.patch("agentwithragv2.FAISS.from_documents")

        rag_obj = agentwithragv2.RAG()
        result = rag_obj.process_multiple_files([mock_file])

        assert result.name == "multi_doc_search"

    def test_directory_creation(self, mocker, mock_uploaded_file):
        mocker.patch("os.path.exists", return_value=False)
        mock_makedirs = mocker.patch("os.makedirs")
        mocker.patch("builtins.open", mocker.mock_open())
        mocker.patch("logic.UnstructuredFileLoader")
        mocker.patch("logic.OpenAIEmbeddings")
        mocker.patch("logic.FAISS.from_documents")

        process_multiple_files([mock_uploaded_file])

        mock_makedirs.assert_called_once_with("temp_docs")