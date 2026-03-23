"""
Simple AI agent instrumented with the NodeLoom SDK.

This agent answers questions using OpenAI and reports all telemetry
(traces, spans, token usage) to your NodeLoom instance.

Usage:
    export OPENAI_API_KEY="sk-..."
    export NODELOOM_API_KEY="sdk_..."
    export NODELOOM_ENDPOINT="https://your-instance.nodeloom.io"  # optional

    pip install nodeloom-sdk openai
    python examples/simple_agent.py "What are AI guardrails?"
"""

import os
import sys
from openai import OpenAI
from nodeloom import NodeLoom, SpanType, TraceStatus

# --- Config ---
NODELOOM_API_KEY = os.environ.get("NODELOOM_API_KEY", "sdk_...")
NODELOOM_ENDPOINT = os.environ.get("NODELOOM_ENDPOINT", "https://api.nodeloom.io")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"
AGENT_NAME = "simple-qa-agent"
AGENT_VERSION = "1.0.0"

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. Answer questions concisely and accurately. "
    "If you don't know the answer, say so."
)


def run_agent(query: str) -> str:
    """Run the agent on a single query, fully instrumented with NodeLoom."""

    # 1. Initialize clients
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    nodeloom = NodeLoom(
        api_key=NODELOOM_API_KEY,
        endpoint=NODELOOM_ENDPOINT,
    )

    # 2. Start a trace for this agent run
    trace = nodeloom.trace(
        AGENT_NAME,
        input={"query": query},
        agent_version=AGENT_VERSION,
    )

    answer = ""

    try:
        # 3. LLM call — wrapped in a span
        with trace.span("openai-chat", type=SpanType.LLM) as span:
            span.set_input({"model": MODEL, "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ]})

            response = openai_client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.3,
            )

            answer = response.choices[0].message.content or ""

            span.set_output({"response": answer})
            span.set_token_usage(
                prompt=response.usage.prompt_tokens,
                completion=response.usage.completion_tokens,
                model=MODEL,
            )

        # 4. End the trace successfully
        trace.end(status=TraceStatus.SUCCESS, output={"answer": answer})

    except Exception as e:
        # 4b. End the trace with error
        trace.end(status=TraceStatus.ERROR, error=str(e))
        raise

    finally:
        # 5. Flush telemetry before exiting
        nodeloom.shutdown()

    return answer


def main():
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable is required")
        sys.exit(1)

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is NodeLoom?"

    print(f"\n  Query: {query}\n")
    answer = run_agent(query)
    print(f"  Answer: {answer}\n")


if __name__ == "__main__":
    main()
