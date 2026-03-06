"""Tests for the LangChain integration callback handler."""

import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from uuid import UUID, uuid4

from nodeloom.client import NodeLoomClient
from nodeloom.queue import TelemetryQueue
from nodeloom.trace import Trace
from nodeloom.span import Span
from nodeloom.types import SpanType, TraceStatus


# We need to mock langchain_core before importing the handler,
# because the import will fail if langchain_core is not installed.
# Instead, we'll mock the imports at the module level.

def _make_mock_client():
    """Create a mock NodeLoomClient that returns real Trace/Span objects."""
    mock_client = MagicMock(spec=NodeLoomClient)
    queue = MagicMock(spec=TelemetryQueue)

    def mock_trace(agent_name, input=None, agent_version=None, metadata=None):
        return Trace(
            agent_name=agent_name,
            queue=queue,
            input_data=input,
            agent_version=agent_version,
        )

    mock_client.trace.side_effect = mock_trace
    return mock_client, queue


class TestNodeLoomCallbackHandlerImport(unittest.TestCase):
    """Test that the handler can be imported when langchain_core is available."""

    @patch.dict("sys.modules", {
        "langchain_core": MagicMock(),
        "langchain_core.callbacks": MagicMock(),
        "langchain_core.outputs": MagicMock(),
        "langchain_core.agents": MagicMock(),
        "langchain_core.documents": MagicMock(),
    })
    def test_import_succeeds_with_langchain(self):
        # Force reimport
        import importlib
        import nodeloom.integrations.langchain as lc_mod
        # Should not raise
        self.assertTrue(hasattr(lc_mod, "NodeLoomCallbackHandler"))


class TestNodeLoomCallbackHandler(unittest.TestCase):
    """Tests for the LangChain callback handler.

    These tests use real Trace/Span objects with a mocked queue to verify
    that the callback handler correctly maps LangChain events to
    NodeLoom telemetry.
    """

    def setUp(self):
        # We need to mock langchain_core to import the handler
        self.lc_callbacks = MagicMock()
        self.lc_outputs = MagicMock()
        self.lc_agents = MagicMock()
        self.lc_documents = MagicMock()

        # Create a mock BaseCallbackHandler that is a real class
        class MockBaseCallbackHandler:
            def __init__(self):
                pass

        self.lc_callbacks.BaseCallbackHandler = MockBaseCallbackHandler

        # Create mock LLMResult
        class MockLLMResult:
            def __init__(self, generations=None, llm_output=None):
                self.generations = generations or []
                self.llm_output = llm_output

        self.MockLLMResult = MockLLMResult

        # Create mock Generation
        class MockGeneration:
            def __init__(self, text=""):
                self.text = text

        self.MockGeneration = MockGeneration

        # Create mock AgentAction
        class MockAgentAction:
            def __init__(self, tool="", tool_input="", log=""):
                self.tool = tool
                self.tool_input = tool_input
                self.log = log

        self.MockAgentAction = MockAgentAction

        # Create mock AgentFinish
        class MockAgentFinish:
            def __init__(self, return_values=None, log=""):
                self.return_values = return_values or {}
                self.log = log

        self.MockAgentFinish = MockAgentFinish

        # Create mock Document
        class MockDocument:
            def __init__(self, page_content="", metadata=None):
                self.page_content = page_content
                self.metadata = metadata or {}

        self.MockDocument = MockDocument

        # Patch the langchain_core modules
        self.patches = {
            "langchain_core": MagicMock(),
            "langchain_core.callbacks": self.lc_callbacks,
            "langchain_core.outputs": self.lc_outputs,
            "langchain_core.agents": self.lc_agents,
            "langchain_core.documents": self.lc_documents,
        }

        self.lc_outputs.LLMResult = self.MockLLMResult
        self.lc_agents.AgentAction = self.MockAgentAction
        self.lc_agents.AgentFinish = self.MockAgentFinish
        self.lc_documents.Document = self.MockDocument

        import sys
        for mod_name, mock_mod in self.patches.items():
            sys.modules[mod_name] = mock_mod

        # Reimport the module to pick up mocks
        import importlib
        import nodeloom.integrations.langchain as lc_mod
        importlib.reload(lc_mod)
        self.NodeLoomCallbackHandler = lc_mod.NodeLoomCallbackHandler

        # Set up client and queue
        self.mock_client, self.mock_queue = _make_mock_client()

    def tearDown(self):
        import sys
        for mod_name in self.patches:
            sys.modules.pop(mod_name, None)

    def _make_handler(self, **kwargs):
        return self.NodeLoomCallbackHandler(self.mock_client, **kwargs)

    # -- Chain tests ---------------------------------------------------------

    def test_chain_start_creates_trace_and_span(self):
        handler = self._make_handler(agent_name="test-agent")
        run_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "MyChain"},
            inputs={"query": "hello"},
            run_id=run_id,
            parent_run_id=None,
        )

        # Should have created a trace
        self.mock_client.trace.assert_called_once_with(
            agent_name="test-agent",
            input={"query": "hello"},
            agent_version=None,
        )
        # Should have registered the span
        self.assertIn(run_id, handler._spans)
        self.assertIn(run_id, handler._traces)

    def test_chain_end_completes_span_and_trace(self):
        handler = self._make_handler()
        run_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "Chain"},
            inputs={"q": "hi"},
            run_id=run_id,
        )

        self.mock_queue.reset_mock()

        handler.on_chain_end(
            outputs={"result": "done"},
            run_id=run_id,
        )

        # Span and trace should be ended
        self.assertNotIn(run_id, handler._spans)
        self.assertNotIn(run_id, handler._traces)

        # Check events emitted: span end + trace end
        calls = self.mock_queue.put.call_args_list
        events = [c[0][0] for c in calls]

        span_events = [e for e in events if e["type"] == "span"]
        trace_end_events = [e for e in events if e["type"] == "trace_end"]

        self.assertEqual(len(span_events), 1)
        self.assertEqual(span_events[0]["status"], "success")

        self.assertEqual(len(trace_end_events), 1)
        self.assertEqual(trace_end_events[0]["status"], "success")

    def test_chain_error(self):
        handler = self._make_handler()
        run_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "Chain"},
            inputs={"q": "hi"},
            run_id=run_id,
        )

        self.mock_queue.reset_mock()

        handler.on_chain_error(
            error=ValueError("something broke"),
            run_id=run_id,
        )

        events = [c[0][0] for c in self.mock_queue.put.call_args_list]
        span_events = [e for e in events if e["type"] == "span"]
        self.assertEqual(len(span_events), 1)
        self.assertEqual(span_events[0]["status"], "error")
        self.assertEqual(span_events[0]["error"], "something broke")

    # -- Nested chain/LLM tests ---------------------------------------------

    def test_nested_chain_and_llm(self):
        handler = self._make_handler()
        chain_run_id = uuid4()
        llm_run_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "AgentChain"},
            inputs={"query": "test"},
            run_id=chain_run_id,
        )

        handler.on_llm_start(
            serialized={"name": "GPT4"},
            prompts=["Hello world"],
            run_id=llm_run_id,
            parent_run_id=chain_run_id,
        )

        self.assertIn(llm_run_id, handler._spans)
        llm_span = handler._spans[llm_run_id]
        chain_span_id = handler._spans.get(chain_run_id)
        if chain_span_id:
            self.assertEqual(llm_span.parent_span_id, chain_span_id.span_id)

        # End LLM
        llm_result = self.MockLLMResult(
            generations=[[self.MockGeneration(text="Hi there")]],
            llm_output={
                "token_usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                },
                "model_name": "gpt-4o",
            },
        )

        self.mock_queue.reset_mock()
        handler.on_llm_end(response=llm_result, run_id=llm_run_id)

        # LLM span should be ended with token usage
        llm_events = [
            c[0][0] for c in self.mock_queue.put.call_args_list
            if c[0][0].get("type") == "span"
        ]
        self.assertEqual(len(llm_events), 1)
        self.assertEqual(llm_events[0]["status"], "success")
        self.assertEqual(llm_events[0]["token_usage"]["prompt_tokens"], 10)
        self.assertEqual(llm_events[0]["token_usage"]["completion_tokens"], 5)
        self.assertEqual(llm_events[0]["token_usage"]["model"], "gpt-4o")

    def test_llm_end_without_token_usage(self):
        handler = self._make_handler()
        chain_run_id = uuid4()
        llm_run_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "Chain"},
            inputs={},
            run_id=chain_run_id,
        )
        handler.on_llm_start(
            serialized={"name": "LLM"},
            prompts=["test"],
            run_id=llm_run_id,
            parent_run_id=chain_run_id,
        )

        self.mock_queue.reset_mock()
        llm_result = self.MockLLMResult(
            generations=[[self.MockGeneration(text="output")]],
            llm_output=None,
        )
        handler.on_llm_end(response=llm_result, run_id=llm_run_id)

        events = [c[0][0] for c in self.mock_queue.put.call_args_list if c[0][0].get("type") == "span"]
        self.assertEqual(len(events), 1)
        # No token_usage key since there was none
        self.assertNotIn("token_usage", events[0])

    # -- LLM error -----------------------------------------------------------

    def test_llm_error(self):
        handler = self._make_handler()
        chain_run_id = uuid4()
        llm_run_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "Chain"},
            inputs={},
            run_id=chain_run_id,
        )
        handler.on_llm_start(
            serialized={"name": "LLM"},
            prompts=["test"],
            run_id=llm_run_id,
            parent_run_id=chain_run_id,
        )

        self.mock_queue.reset_mock()
        handler.on_llm_error(error=RuntimeError("rate limit"), run_id=llm_run_id)

        events = [c[0][0] for c in self.mock_queue.put.call_args_list if c[0][0].get("type") == "span"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "error")
        self.assertEqual(events[0]["error"], "rate limit")

    # -- Tool tests ----------------------------------------------------------

    def test_tool_start_and_end(self):
        handler = self._make_handler()
        chain_run_id = uuid4()
        tool_run_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "Agent"},
            inputs={"query": "weather"},
            run_id=chain_run_id,
        )
        handler.on_tool_start(
            serialized={"name": "weather_api"},
            input_str="What is the weather?",
            run_id=tool_run_id,
            parent_run_id=chain_run_id,
        )

        self.assertIn(tool_run_id, handler._spans)

        self.mock_queue.reset_mock()
        handler.on_tool_end(output="72F and sunny", run_id=tool_run_id)

        events = [c[0][0] for c in self.mock_queue.put.call_args_list if c[0][0].get("type") == "span"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["span_type"], "tool")
        self.assertEqual(events[0]["output"], {"output": "72F and sunny"})

    def test_tool_error(self):
        handler = self._make_handler()
        chain_run_id = uuid4()
        tool_run_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "Agent"},
            inputs={},
            run_id=chain_run_id,
        )
        handler.on_tool_start(
            serialized={"name": "calculator"},
            input_str="1/0",
            run_id=tool_run_id,
            parent_run_id=chain_run_id,
        )

        self.mock_queue.reset_mock()
        handler.on_tool_error(error=ZeroDivisionError("division by zero"), run_id=tool_run_id)

        events = [c[0][0] for c in self.mock_queue.put.call_args_list if c[0][0].get("type") == "span"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "error")

    # -- Retriever tests -----------------------------------------------------

    def test_retriever_start_and_end(self):
        handler = self._make_handler()
        chain_run_id = uuid4()
        ret_run_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "RAGChain"},
            inputs={"query": "docs"},
            run_id=chain_run_id,
        )
        handler.on_retriever_start(
            serialized={"name": "VectorStore"},
            query="relevant docs",
            run_id=ret_run_id,
            parent_run_id=chain_run_id,
        )

        self.assertIn(ret_run_id, handler._spans)
        ret_span = handler._spans[ret_run_id]
        self.assertEqual(ret_span._span_type, SpanType.RETRIEVAL)

        self.mock_queue.reset_mock()

        docs = [
            self.MockDocument(page_content="Doc 1", metadata={"source": "a"}),
            self.MockDocument(page_content="Doc 2", metadata={"source": "b"}),
        ]
        handler.on_retriever_end(documents=docs, run_id=ret_run_id)

        events = [c[0][0] for c in self.mock_queue.put.call_args_list if c[0][0].get("type") == "span"]
        self.assertEqual(len(events), 1)
        output = events[0]["output"]
        self.assertEqual(len(output["documents"]), 2)
        self.assertEqual(output["documents"][0]["page_content"], "Doc 1")

    def test_retriever_error(self):
        handler = self._make_handler()
        chain_run_id = uuid4()
        ret_run_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "RAG"},
            inputs={},
            run_id=chain_run_id,
        )
        handler.on_retriever_start(
            serialized={"name": "VectorStore"},
            query="query",
            run_id=ret_run_id,
            parent_run_id=chain_run_id,
        )

        self.mock_queue.reset_mock()
        handler.on_retriever_error(error=ConnectionError("db down"), run_id=ret_run_id)

        events = [c[0][0] for c in self.mock_queue.put.call_args_list if c[0][0].get("type") == "span"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["status"], "error")

    # -- Agent action/finish -------------------------------------------------

    def test_agent_action_emits_event(self):
        handler = self._make_handler()
        chain_run_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "Agent"},
            inputs={"query": "test"},
            run_id=chain_run_id,
        )

        self.mock_queue.reset_mock()

        action = self.MockAgentAction(
            tool="search",
            tool_input={"query": "AI news"},
            log="Searching for AI news",
        )
        handler.on_agent_action(action=action, run_id=chain_run_id)

        events = [c[0][0] for c in self.mock_queue.put.call_args_list if c[0][0].get("type") == "event"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["name"], "agent_action")
        self.assertEqual(events[0]["data"]["tool"], "search")

    def test_agent_finish_emits_event(self):
        handler = self._make_handler()
        chain_run_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "Agent"},
            inputs={},
            run_id=chain_run_id,
        )

        self.mock_queue.reset_mock()

        finish = self.MockAgentFinish(
            return_values={"output": "done"},
            log="Agent finished",
        )
        handler.on_agent_finish(finish=finish, run_id=chain_run_id)

        events = [c[0][0] for c in self.mock_queue.put.call_args_list if c[0][0].get("type") == "event"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["name"], "agent_finish")

    # -- Edge cases ----------------------------------------------------------

    def test_end_span_for_unknown_run_id(self):
        handler = self._make_handler()
        unknown_id = uuid4()

        # Should not raise
        handler.on_chain_end(outputs={"x": 1}, run_id=unknown_id)
        handler.on_llm_end(
            response=self.MockLLMResult(generations=[], llm_output=None),
            run_id=unknown_id,
        )
        handler.on_tool_end(output="x", run_id=unknown_id)

    def test_serialized_name_fallback(self):
        handler = self._make_handler()
        run_id = uuid4()

        # No "name" key, should fall back to last element of "id"
        handler.on_chain_start(
            serialized={"id": ["langchain", "chains", "MyCustomChain"]},
            inputs={},
            run_id=run_id,
        )

        span = handler._spans.get(run_id)
        self.assertIsNotNone(span)
        self.assertEqual(span.name, "MyCustomChain")

    def test_no_op_callbacks(self):
        handler = self._make_handler()
        # These should not raise
        handler.on_text("some text")
        handler.on_llm_new_token("tok")

    def test_handler_reusable_across_invocations(self):
        handler = self._make_handler()

        # First invocation
        run_id_1 = uuid4()
        handler.on_chain_start(
            serialized={"name": "Chain"},
            inputs={"q": "first"},
            run_id=run_id_1,
        )
        handler.on_chain_end(outputs={"r": "1"}, run_id=run_id_1)

        # Second invocation
        run_id_2 = uuid4()
        handler.on_chain_start(
            serialized={"name": "Chain"},
            inputs={"q": "second"},
            run_id=run_id_2,
        )
        handler.on_chain_end(outputs={"r": "2"}, run_id=run_id_2)

        # Both traces should have been created
        self.assertEqual(self.mock_client.trace.call_count, 2)

    def test_token_usage_alternative_keys(self):
        """Test that alternative token usage keys (input_tokens, output_tokens) are handled."""
        handler = self._make_handler()
        chain_run_id = uuid4()
        llm_run_id = uuid4()

        handler.on_chain_start(
            serialized={"name": "Chain"},
            inputs={},
            run_id=chain_run_id,
        )
        handler.on_llm_start(
            serialized={"name": "Claude"},
            prompts=["test"],
            run_id=llm_run_id,
            parent_run_id=chain_run_id,
        )

        self.mock_queue.reset_mock()

        # Anthropic-style token keys
        llm_result = self.MockLLMResult(
            generations=[[self.MockGeneration(text="response")]],
            llm_output={
                "usage": {
                    "input_tokens": 15,
                    "output_tokens": 25,
                },
                "model": "claude-3-opus",
            },
        )
        handler.on_llm_end(response=llm_result, run_id=llm_run_id)

        events = [c[0][0] for c in self.mock_queue.put.call_args_list if c[0][0].get("type") == "span"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["token_usage"]["prompt_tokens"], 15)
        self.assertEqual(events[0]["token_usage"]["completion_tokens"], 25)
        self.assertEqual(events[0]["token_usage"]["model"], "claude-3-opus")


if __name__ == "__main__":
    unittest.main()
