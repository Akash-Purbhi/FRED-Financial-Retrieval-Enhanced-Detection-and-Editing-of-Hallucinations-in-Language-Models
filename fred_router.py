import ollama
import json
import re

# ============================================================
# FRED Router v3 — JSON Extraction + XML Tagging Pipeline
# ============================================================
# Step 1: Model extracts structured JSON with context truth vs response error
# Step 2: Python validates the JSON
# Step 3: Model produces the final XML-tagged output
# ============================================================

def run_fred_json_pipeline(context_and_question_and_response: str):
    """
    Two-stage pipeline:
      Stage 1 — Structured JSON extraction (what's wrong?)
      Stage 2 — XML-tagged correction (fix it in-place)
    """

    # ---- STAGE 1: JSON Extraction ----
    json_messages = [
        {
            'role': 'system',
            'content': (
                'You are a financial hallucination detection assistant. '
                'Your job is to compare the Response against the Context and identify numerical errors. '
                'Output ONLY a JSON object with these fields:\n'
                '  "has_error": true/false,\n'
                '  "errors": [{"response_value": "...", "context_truth": "..."}]\n'
                'Do not output anything else.'
            )
        },
        # Few-shot example 1
        {
            'role': 'user',
            'content': '''Context: [Doc 1] Total assets for 2021 were $500 million.
Question: What were the total assets in 2021?
Response: Total assets for 2021 were $400 million.'''
        },
        {
            'role': 'assistant',
            'content': '{"has_error": true, "errors": [{"response_value": "$400", "context_truth": "$500"}]}'
        },
        # Few-shot example 2
        {
            'role': 'user',
            'content': '''Context: [Doc 1] Revenue was $1.2 billion in 2022.
Question: What was the revenue in 2022?
Response: Revenue was $1.2 billion in 2022.'''
        },
        {
            'role': 'assistant',
            'content': '{"has_error": false, "errors": []}'
        },
        # Actual query
        {
            'role': 'user',
            'content': context_and_question_and_response
        }
    ]

    print("Stage 1: Extracting errors via JSON...")
    json_response = ollama.chat(
        model='fred_qwen',
        messages=json_messages,
        options={'temperature': 0}
    )
    raw_json = json_response['message']['content'].strip()
    print(f"  Raw JSON output: {raw_json}")

    # ---- VALIDATE JSON ----
    try:
        # Try to extract JSON from the response (model might add extra text)
        json_match = re.search(r'\{.*\}', raw_json, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            parsed = json.loads(raw_json)

        has_error = parsed.get('has_error', False)
        errors = parsed.get('errors', [])

        print(f"  Parsed successfully: has_error={has_error}, errors={errors}")

        if not has_error or not errors:
            print("\n  No errors detected — response appears correct.")
            return

        # Validate that each error has both fields
        for err in errors:
            if 'response_value' not in err or 'context_truth' not in err:
                print(f"  WARNING: Malformed error entry: {err}")
                return

    except json.JSONDecodeError as e:
        print(f"  ERROR: Could not parse JSON — {e}")
        print("  Falling back to direct tagging...")
        run_direct_tagging(context_and_question_and_response)
        return

    # ---- STAGE 2: XML Tagging ----
    tag_messages = [
        {
            'role': 'system',
            'content': (
                'You are a financial hallucination detection assistant. '
                'Reproduce the response exactly, wrapping any numerical errors '
                'in <numerical><delete>wrong</delete><mark>correct</mark></numerical> tags.'
            )
        },
        # Few-shot example
        {
            'role': 'user',
            'content': '''Context: [Doc 1] Total assets for 2021 were $500 million.
Question: What were the total assets in 2021?
Response: Total assets for 2021 were $400 million.'''
        },
        {
            'role': 'assistant',
            'content': 'Total assets for 2021 were <numerical><delete>$400</delete><mark>$500</mark></numerical> million.'
        },
        # Actual query
        {
            'role': 'user',
            'content': context_and_question_and_response
        }
    ]

    print("\nStage 2: Generating XML-tagged output...")
    tag_response = ollama.chat(
        model='fred_qwen',
        messages=tag_messages,
        options={'temperature': 0}
    )
    final_output = tag_response['message']['content']

    # ---- STAGE 3: Python Validation ----
    print("\nStage 3: Validating output...")
    validation_passed = True
    for err in errors:
        if err['response_value'] not in final_output and f"<delete>{err['response_value']}</delete>" not in final_output:
            print(f"  WARNING: Expected error value '{err['response_value']}' not found in output")
            validation_passed = False
        if err['context_truth'] not in final_output and f"<mark>{err['context_truth']}</mark>" not in final_output:
            print(f"  WARNING: Expected correction '{err['context_truth']}' not found in output")
            validation_passed = False

    if validation_passed:
        print("  ✓ Validation passed — all identified errors are tagged.")
    else:
        print("  ✗ Validation issues detected — review output manually.")

    print("\n" + "=" * 50)
    print("FRED Final Output:")
    print("=" * 50)
    print(final_output)


def run_direct_tagging(context_and_question_and_response: str):
    """Fallback: direct tagging without JSON extraction."""
    messages = [
        {
            'role': 'system',
            'content': (
                'You are a financial hallucination detection assistant. '
                'Reproduce the response exactly, wrapping any numerical errors '
                'in <numerical><delete>wrong</delete><mark>correct</mark></numerical> tags.'
            )
        },
        {
            'role': 'user',
            'content': '''Context: [Doc 1] Total assets for 2021 were $500 million.
Question: What were the total assets in 2021?
Response: Total assets for 2021 were $400 million.'''
        },
        {
            'role': 'assistant',
            'content': 'Total assets for 2021 were <numerical><delete>$400</delete><mark>$500</mark></numerical> million.'
        },
        {
            'role': 'user',
            'content': context_and_question_and_response
        }
    ]

    response = ollama.chat(
        model='fred_qwen',
        messages=messages,
        options={'temperature': 0}
    )
    print("\n--- FRED Direct Output ---")
    print(response['message']['content'])


# ============================================================
# TEST CASE
# ============================================================
if __name__ == '__main__':
    test_input = '''Context: [Doc 1] leases , was $ 92 million , $ 80 million , and $ 72 million in 2002 , 2001 , and 2000 , respectively . future minimum lease payments under noncancelable operating leases having remaining terms in excess of one year as of september 28 , 2002 , are as follows ( in millions ) :
[Doc 2] [["2003", "$ 83"], ["2004", "78"], ["2005", "66"], ["2006", "55"], ["2007", "42"], ["later years", "140"], ["total minimum lease payments", "$ 464"]]
Question: what percentage of total minimum lease payments is due in 2004?
Response: To find the percentage of total minimum lease payments due in 2004, we first need to calculate the total minimum lease payments from 2003 to later years. Total from 2003 to later years: $83 + $78 + $66 + $55 + $42 + $140 = $464. Percentage due in 2004 = ($88 / $464) * 100%.'''

    run_fred_json_pipeline(test_input)
