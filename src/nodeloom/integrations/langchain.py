"""LangChain integration for NodeLoom telemetry.

Provides a callback handler that automatically maps LangChain's
chain/LLM/tool/retriever lifecycle events to NodeLoom traces and spans.

Usage::

    from nodeloom import NodeLoom
    from nodeloom.integrations.langchain import NodeLoomCallbackHandler

    client = NodeLoom(api_key="sdk_...")
    handler = NodeLoomCallbackHandler(client)
    chain.invoke(input, config={"callbacks": [handler]})

Requires ``langchain-core`` to be installed. If it is not available,
importing this module will raise an ImportError with a clear message.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID

from nodeloom.client import NodeLoomClient
from nodeloom.span import Span
from nodeloom.trace import Trace
from nodeloom.types import SpanType, TraceStatus

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.documents import Document
except ImportError:
    raise ImportError(
        "langchain-core is required for the LangChain integration. "
        "Install it with: pip install nodeloom-sdk[langchain]"
    )

logger = logging.getLogger("nodeloom.integrations.langchain")


class NodeLoomCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that sends telemetry to NodeLoom.

    This handler creates a single trace per invocation and maps
    LangChain's nested run hierarchy to NodeLoom spans. Each run_id
    from LangChain becomes a span; parent_run_id links are preserved.

    The handler is designed to be reusable across multiple invocations.
    Each top-level chain_start (with no parent_run_id) creates a new
    trace. Subsequent events within that hierarchy are attached as spans.
    """

    def __init__(
        self,
        client: NodeLoomClient,
        agent_name: str = "langchain-agent",
        agent_version: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._client = client
        self._agent_name = agent_name
        self._agent_version = agent_version

        # Maps run_id -> Span
        self._spans: Dict[UUID, Span] = {}
        # Maps run_id -> Trace (only for root runs)
        self._traces: Dict[UUID, Trace] = {}
        # Maps run_id -> parent_run_id for hierarchy lookups
        self._parents: Dict[UUID, Optional[UUID]] = {}

    def _get_trace_for_run(self, run_id: UUID) -> Optional[Trace]:
        """Walk up the parent chain to find the root trace."""
        current = run_id
        while current is not None:
            if current in self._traces:
                return self._traces[current]
            current = self._parents.get(current)
        return None

    def _get_parent_span_id(self, parent_run_id: Optional[UUID]) -> Optional[str]:
        """Resolve the NodeLoom span_id for a LangChain parent_run_id."""
        if parent_run_id is None:
            return None
        parent_span = self._spans.get(parent_run_id)
        if parent_span is not None:
            return parent_span.span_id
        return None

    def _start_span(
        self,
        run_id: UUID,
        parent_run_id: Optional[UUID],
        name: str,
        span_type: SpanType,
        input_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Span]:
        """Create and register a new span, or a new trace if this is a root run."""
        self._parents[run_id] = parent_run_id

        # Root run: create a new trace
        if parent_run_id is None:
            trace = self._client.trace(
                agent_name=self._agent_name,
                input=input_data,
                agent_version=self._agent_version,
            )
            self._traces[run_id] = trace
            span = trace.span(name=name, type=span_type)
            if input_data:
                span.set_input(input_data)
            self._spans[run_id] = span
            return span

        # Child run: find the parent trace and create a span
        trace = self._get_trace_for_run(run_id)
        if trace is None:
            logger.warning(
                "Could not find trace for run %s (parent: %s). Skipping span.",
                run_id,
                parent_run_id,
            )
            return None

        parent_span_id = self._get_parent_span_id(parent_run_id)
        span = trace.span(
            name=name,
            type=span_type,
            parent_span_id=parent_span_id,
        )
        if input_data:
            span.set_input(input_data)
        self._spans[run_id] = span
        return span

    def _end_span(
        self,
        run_id: UUID,
        output: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """End a span and, if this is a root run, end the trace too."""
        span = self._spans.pop(run_id, None)
        if span is None:
            return

        if error is not None:
            span.set_error(error)
            span.end(status=TraceStatus.ERROR, output=output)
        else:
            span.end(status=TraceStatus.SUCCESS, output=output)

        # If this is a root run, also end the trace
        trace = self._traces.pop(run_id, None)
        if trace is not None:
            status = TraceStatus.ERROR if error else TraceStatus.SUCCESS
            trace.end(status=status, output=output, error=error)

        self._parents.pop(run_id, None)

    # -- Chain callbacks -----------------------------------------------------

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name") or serialized.get("id", ["Chain"])[-1]
        self._start_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name,
            span_type=SpanType.CHAIN,
            input_data=inputs if isinstance(inputs, dict) else {"input": inputs},
        )

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        output = outputs if isinstance(outputs, dict) else {"output": outputs}
        self._end_span(run_id=run_id, output=output)

    def on_chain_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._end_span(run_id=run_id, error=str(error))

    # -- LLM callbacks -------------------------------------------------------

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name") or serialized.get("id", ["LLM"])[-1]
        self._start_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name,
            span_type=SpanType.LLM,
            input_data={"prompts": prompts},
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        span = self._spans.get(run_id)
        if span is not None:
            # Extract token usage from llm_output if available
            llm_output = response.llm_output or {}
            token_usage = llm_output.get("token_usage") or llm_output.get(
                "usage", {}
            )
            if token_usage:
                prompt_tokens = token_usage.get(
                    "prompt_tokens", token_usage.get("input_tokens", 0)
                )
                completion_tokens = token_usage.get(
                    "completion_tokens", token_usage.get("output_tokens", 0)
                )
                model = llm_output.get("model_name") or llm_output.get("model", "")
                span.set_token_usage(
                    prompt=prompt_tokens,
                    completion=completion_tokens,
                    model=model if model else None,
                )

        output_texts = []
        for gen_list in response.generations:
            for gen in gen_list:
                output_texts.append(gen.text)

        self._end_span(run_id=run_id, output={"generations": output_texts})

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._end_span(run_id=run_id, error=str(error))

    # -- Tool callbacks ------------------------------------------------------

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "Tool")
        self._start_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name,
            span_type=SpanType.TOOL,
            input_data={"input": input_str},
        )

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._end_span(run_id=run_id, output={"output": output})

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._end_span(run_id=run_id, error=str(error))

    # -- Retriever callbacks -------------------------------------------------

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name") or serialized.get("id", ["Retriever"])[-1]
        self._start_span(
            run_id=run_id,
            parent_run_id=parent_run_id,
            name=name,
            span_type=SpanType.RETRIEVAL,
            input_data={"query": query},
        )

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        docs = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in documents
        ]
        self._end_span(run_id=run_id, output={"documents": docs})

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._end_span(run_id=run_id, error=str(error))

    # -- Agent callbacks (mapped to chain-like spans) ------------------------

    def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Log agent action as an event on the current trace."""
        trace = self._get_trace_for_run(run_id)
        if trace is not None:
            trace.event(
                name="agent_action",
                level="info",
                data={
                    "tool": action.tool,
                    "tool_input": (
                        action.tool_input
                        if isinstance(action.tool_input, dict)
                        else {"input": action.tool_input}
                    ),
                    "log": action.log,
                },
            )

    def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Log agent finish as an event on the current trace."""
        trace = self._get_trace_for_run(run_id)
        if trace is not None:
            trace.event(
                name="agent_finish",
                level="info",
                data={
                    "output": finish.return_values,
                    "log": finish.log,
                },
            )

    # -- Ignored callbacks ---------------------------------------------------

    def on_text(self, text: str, **kwargs: Any) -> None:
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        pass
