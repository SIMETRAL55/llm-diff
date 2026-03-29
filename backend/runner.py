import logging
import litellm

logger = logging.getLogger(__name__)


def run_test_cases(config: dict) -> list[dict]:
    """Run v1 and v2 prompts for each test case, return outputs.

    Args:
        config: dict with keys "model" and "test_cases"

    Returns:
        List of dicts: {id, input, output_v1, output_v2}
    """
    model = config["model"]
    results = []

    for tc in config["test_cases"]:
        tc_id = tc.get("id", "unknown")
        input_text = tc.get("input", "")
        context = tc.get("context", "")

        substitutions = {"input": input_text, "context": context}

        try:
            prompt_v1 = tc["prompt_v1"].format_map(substitutions)
            prompt_v2 = tc["prompt_v2"].format_map(substitutions)

            response_v1 = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt_v1}],
            )
            output_v1 = response_v1.choices[0].message.content

            response_v2 = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": prompt_v2}],
            )
            output_v2 = response_v2.choices[0].message.content

            results.append({
                "id": tc_id,
                "input": input_text,
                "output_v1": output_v1,
                "output_v2": output_v2,
            })

        except Exception as e:
            logger.error(f"Test case {tc_id} failed: {e}")
            results.append({
                "id": tc_id,
                "input": input_text,
                "output_v1": f"ERROR: {e}",
                "output_v2": f"ERROR: {e}",
            })

    return results
