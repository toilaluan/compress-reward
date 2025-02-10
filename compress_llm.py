from openai import OpenAI
import re

PROMPT_TEMPLATE = """
You are an advanced text compression expert. Transform the text in <original-text> tag into its most concise version while preserving intent, crucial details, tone, and any elements requiring exact wording (e.g., translations, technical/legal terms). Adhere to these steps:

1. Context Analysis
   - Determine purpose, tone, style
   - Identify protected vs. compressible elements

2. Special Data Enumeration
   - List all events, dates, numbers

3. Modular Compression
   a) Hierarchical Compression
      - Remove redundancies
      - Simplify structures
      - Use abbreviations/symbols (informal only)
      - Reorganize for clarity
   b) Context-Sensitive Optimization
      - Keep exact text for technical/legal/medical/cultural specifics
      - Preserve numeric data/equations verbatim
   c) Visual Compression
      - Employ bullet points, tables, mathematical notation (e.g. 5×/week), and common symbols (→ for "leads to")
   d) Intent Preservation
      - Maintain all action verbs
      - Keep conditional logic
      - Retain cause-effect chains

4. Validation
   - Confirm factual accuracy, logical flow, intent, and action steps remain intact
   - Ensure critical information can be restored if needed

Wrap your analysis in <compression_analysis> tags, covering:
1. Purpose, tone, style
2. Protected vs. compressible elements
3. Special data (events, dates, numbers)
4. Before/after text comparisons
5. Word count reduction % and preserved elements

Then provide:

<optimization_report>
1. Context Type
2. Protected Elements
3. Special Data
4. Compression Map
5. Validation Metrics
</optimization_report>

<optimized_text>
[Output the final compressed version here]
</optimized_text>

Process above instructions based on the following original text:
<original-text>
{INPUT_TEXT}
</original-text>
"""


def compress_by_llm(
    text: str,
    openai_client: OpenAI,
    model_name: str = "gpt-4o-mini",
    max_retries: int = 3,
):
    for attempt in range(max_retries):
        try:
            prompt = PROMPT_TEMPLATE.format(INPUT_TEXT=text)
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=8000,
            )
            print(response)
            # Extract the optimized text from the response
            completion = response.choices[0].message.content

            # Find content between optimized_text tags
            match = re.search(
                r"<optimized_text>(.*?)</optimized_text>", completion, re.DOTALL
            )
            if not match:
                raise ValueError("Could not find optimized text content between tags")

            return match.group(1).strip()

        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                raise Exception(
                    f"Failed after {max_retries} attempts. Last error: {str(e)}"
                )
            print(f"Attempt {attempt + 1} failed. Retrying... Error: {str(e)}")
