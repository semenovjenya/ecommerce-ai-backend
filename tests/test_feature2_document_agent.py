import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from agi_ecom_app.api import app
from agi_ecom_app.document_import.contract_airflow import trigger_contract_airflow_worker
from agi_ecom_app.document_import.document_agent import ContractReference, run_feature2_document_agent
from agi_ecom_app.document_import.models import ContractAirflowWebhookResponse, Feature2AgentResponse


class Feature2ContractAirflowTests(unittest.TestCase):
    @patch("agi_ecom_app.document_import.contract_airflow.httpx.post")
    @patch("agi_ecom_app.document_import.contract_airflow.get_settings")
    def test_trigger_contract_airflow_worker_posts_dag_run(self, mock_get_settings, mock_post):
        mock_get_settings.return_value = SimpleNamespace(
            airflow_base_url="http://airflow.local",
            airflow_username="admin",
            airflow_password="admin",
            airflow_contract_dag_id="contract_document_ingest",
        )
        mock_response = mock_post.return_value
        mock_response.json.return_value = {"dag_run_id": "run-1"}
        mock_response.raise_for_status.return_value = None

        response = trigger_contract_airflow_worker(
            slot=1,
            filename="dogovor_postavki_v1.docx",
            content=b"docx-bytes",
        )

        self.assertEqual(response.status, "queued")
        self.assertEqual(response.dagRunId, "run-1")
        self.assertEqual(response.documentId, "dogovor_postavki_v1")
        call_kwargs = mock_post.call_args.kwargs
        self.assertEqual(call_kwargs["auth"], ("admin", "admin"))
        self.assertEqual(call_kwargs["json"]["conf"]["filename"], "dogovor_postavki_v1.docx")
        self.assertEqual(call_kwargs["json"]["conf"]["slot"], 1)
        self.assertIn("content_base64", call_kwargs["json"]["conf"])

    def test_trigger_contract_airflow_worker_rejects_non_docx(self):
        with self.assertRaises(ValueError):
            trigger_contract_airflow_worker(slot=1, filename="contract.pdf", content=b"pdf")


class Feature2DocumentAgentServiceTests(unittest.TestCase):
    @patch("agi_ecom_app.document_import.document_agent.get_active_llm_runtime")
    @patch("agi_ecom_app.document_import.document_agent.get_dalkos_contract_context")
    @patch("agi_ecom_app.document_import.document_agent.get_settings")
    def test_run_feature2_document_agent_uses_only_dalkos_context(
        self,
        mock_get_settings,
        mock_get_context,
        mock_get_runtime,
    ):
        mock_get_settings.return_value = SimpleNamespace(
            openai_api_key="test-key",
            openai_model="gpt-test",
        )
        mock_get_context.return_value = {
            "documents": [
                {
                    "slot": 1,
                    "document_id": "dogovor_postavki_v1",
                    "source_file_basename": "dogovor_postavki_v1.docx",
                }
            ],
            "chunks": [{"document_id": "dogovor_postavki_v1", "content": "Текст из БД Далкос"}],
            "risks": [],
            "tables": [],
        }
        mock_client = SimpleNamespace(responses=SimpleNamespace(create=Mock()))
        mock_get_runtime.return_value = SimpleNamespace(
            sync_client=mock_client,
            model_name="gpt-test",
        )
        mock_client.responses.create.return_value = SimpleNamespace(output_text="Ответ агента")

        response = run_feature2_document_agent(
            message="Проверь документ",
            document_refs=[ContractReference(slot=1, filename="dogovor_postavki_v1.docx")],
        )

        self.assertEqual(response.answer, "Ответ агента")
        self.assertEqual(response.model, "gpt-test")
        self.assertEqual(response.documents[0].documentId, "dogovor_postavki_v1")
        mock_get_runtime.assert_called_once()
        mock_get_context.assert_called_once()
        call_kwargs = mock_client.responses.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "gpt-test")
        self.assertIn("Текст из БД Далкос", call_kwargs["input"])


class Feature2DocumentAgentApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    @patch("agi_ecom_app.document_import.router.run_feature2_document_agent")
    def test_agent_endpoint_accepts_two_document_names_without_file_content(self, mock_run_agent):
        mock_run_agent.return_value = Feature2AgentResponse(
            answer="ok",
            model="gpt-test",
            documents=[
                {
                    "slot": 1,
                    "filename": "one.docx",
                    "sizeBytes": 0,
                    "documentId": "one",
                },
                {
                    "slot": 2,
                    "filename": "two.docx",
                    "sizeBytes": 0,
                    "documentId": "two",
                },
            ],
        )

        response = self.client.post(
            "/v1/feature-2/agent",
            data={
                "message": "Проверь",
                "document_1_name": "one.docx",
                "document_2_name": "two.docx",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "ok")
        call_kwargs = mock_run_agent.call_args.kwargs
        self.assertEqual(call_kwargs["message"], "Проверь")
        self.assertEqual([document.slot for document in call_kwargs["document_refs"]], [1, 2])
        self.assertEqual([document.filename for document in call_kwargs["document_refs"]], ["one.docx", "two.docx"])

    @patch("agi_ecom_app.document_import.router.trigger_contract_airflow_worker")
    def test_contract_airflow_webhook_accepts_upload(self, mock_trigger):
        mock_trigger.return_value = ContractAirflowWebhookResponse(
            slot=1,
            filename="one.docx",
            sizeBytes=5,
            dagId="contract_document_ingest",
            dagRunId="run-1",
            documentId="one",
            contentSha256="abc",
        )

        response = self.client.post(
            "/v1/feature-2/contracts/airflow-webhook",
            data={"slot": "1"},
            files={"file": ("one.docx", b"bytes", "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["dagRunId"], "run-1")
        call_kwargs = mock_trigger.call_args.kwargs
        self.assertEqual(call_kwargs["slot"], 1)
        self.assertEqual(call_kwargs["filename"], "one.docx")
        self.assertEqual(call_kwargs["content"], b"bytes")


if __name__ == "__main__":
    unittest.main()
