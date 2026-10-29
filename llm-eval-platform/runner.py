"""
runner.py — Sends test prompts to Ollama and records responses.

WHAT IT DOES:
  For each TestCase in the test suite:
  1. Sends the prompt to the specified Ollama model
  2. Records the response text
  3. Measures latency (milliseconds)
  4. Counts tokens used
  5. Returns an LLMResponse object

WHY MEASURE LATENCY HERE:
  Latency must be measured around the actual HTTP call — not inside it.
  We use time.perf_counter() which is more precise than time.time()
  for short durations (sub-second measurements).

  time.time()        → wall clock, OS-dependent precision (~10ms)
  time.perf_counter()→ high-resolution timer, ~nanosecond precision
  For measuring 200ms-5000ms responses, perf_counter is the right choice.

LINE BY LINE:
  def run_test_case(model, test_case)
    start = time.perf_counter()
                         → record start time before the HTTP call
    resp = requests.post(...)
                         → the actual LLM call (blocks until response)
    latency_ms = (time.perf_counter() - start) * 1000
                         → end - start = seconds, × 1000 = milliseconds
                         → stored in LLMResponse.latency_ms

  resp.json()["eval_count"]
                         → Ollama returns token counts in the response:
                           "prompt_eval_count" = tokens in the prompt
                           "eval_count" = tokens in the completion
                         → .get(..., 0) = safe access, 0 if key missing

  def run_suite(model, test_cases, progress_callback)
    progress_callback    → optional function called after each test
                           used by main.py to show a progress bar
                           signature: callback(current, total, test_case)

  [run_test_case(model, tc) for tc in test_cases]
                         → list comprehension — runs all test cases
                         → could be parallelised with ThreadPoolExecutor
                           for faster evaluation of large test suites

  except Exception as e  → catches any error (connection, timeout, etc.)
                           returns a placeholder LLMResponse with error
                           so one failed test doesn't stop the whole suite
"""

import time
import requests
from typing import List, Callable, Optional
from rich.console import Console
from models import TestCase, LLMResponse
from config import cfg

console = Console()


def run_test_case(model: str, test_case: TestCase) -> LLMResponse:
    """
    Send one test prompt to Ollama and return the response.

    Measures latency precisely using perf_counter.
    Captures token usage from Ollama's response metadata.

    Args:
        model:     Ollama model name (e.g. "llama3.1:70b")
        test_case: TestCase with the prompt to send.

    Returns:
        LLMResponse with text, latency, and token counts.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": test_case.prompt}],
        "stream": False,
        "options": {"temperature": 0},   # deterministic for consistent eval
    }

    # Record start time with high-precision counter
    start = time.perf_counter()

    try:
        resp = requests.post(
            f"{cfg.OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=cfg.TIMEOUT,
        )
        resp.raise_for_status()

        # Calculate latency in milliseconds
        latency_ms = (time.perf_counter() - start) * 1000

        data = resp.json()
        response_text = data["message"]["content"].strip()

        # Extract token counts from Ollama metadata
        # prompt_eval_count = tokens in the prompt
        # eval_count        = tokens in the completion
        prompt_tokens     = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return LLMResponse(
            model=model,
            prompt=test_case.prompt,
            response=response_text,
            latency_ms=round(latency_ms, 2),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    except requests.exceptions.ConnectionError:
        latency_ms = (time.perf_counter() - start) * 1000
        console.print(f"[red]Cannot connect to Ollama — is it running?[/red]")
        return LLMResponse(
            model=model,
            prompt=test_case.prompt,
            response="ERROR: Cannot connect to Ollama",
            latency_ms=round(latency_ms, 2),
        )

    except requests.exceptions.Timeout:
        latency_ms = (time.perf_counter() - start) * 1000
        return LLMResponse(
            model=model,
            prompt=test_case.prompt,
            response=f"ERROR: Timed out after {cfg.TIMEOUT}s",
            latency_ms=round(latency_ms, 2),
        )

    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return LLMResponse(
            model=model,
            prompt=test_case.prompt,
            response=f"ERROR: {e}",
            latency_ms=round(latency_ms, 2),
        )


def run_suite(
    model: str,
    test_cases: List[TestCase],
    progress_callback: Optional[Callable] = None,
) -> List[LLMResponse]:
    """
    Run all test cases against one model.

    Args:
        model:             Ollama model to test.
        test_cases:        List of TestCase objects to run.
        progress_callback: Optional function(current, total, test_case) for progress.

    Returns:
        List of LLMResponse — one per test case, in order.
    """
    responses = []

    for i, tc in enumerate(test_cases, 1):
        if progress_callback:
            progress_callback(i, len(test_cases), tc)

        response = run_test_case(model, tc)
        responses.append(response)

        console.print(
            f"  [dim][{i}/{len(test_cases)}] {tc.prompt[:50]}... "
            f"→ {response.latency_ms:.0f}ms[/dim]"
        )

    return responses
