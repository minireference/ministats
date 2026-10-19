#!/usr/bin/env python3
"""
Convert the docs source noteboks in `docs/_srcnbs/` to `.md` files in `docs/`.
Requires nbformat; does not execute notebook code.
Saved PNG/SVG plots are exported to docs/assets/<notebook-name>/.
"""
import base64
import glob
import os
import re

import nbformat


IMAGE_MIMETYPES = ("image/png", "image/svg+xml")


def outputs_text(outputs):
    """Collect saved text in order, keeping partial stream writes together."""
    text = ""
    for output in outputs:
        kind = output["output_type"]
        if kind == "stream":  # stdout or stderr
            text += output["text"]
            continue
        if kind in ("execute_result", "display_data"):
            data = output["data"]
            if any(mime in data for mime in IMAGE_MIMETYPES):
                continue  # The image replaces its plain-text representation.
            chunk = data.get("text/plain", "[Non-text output omitted]")
        elif kind == "error":
            chunk = "\n".join(output["traceback"])
            if not chunk:
                chunk = f"{output['ename']}: {output['evalue']}"
        else:
            continue

        if not chunk:
            continue
        if text and not text.endswith("\n"):
            text += "\n"
        text += chunk.rstrip("\n") + "\n"

    # Remove terminal colour codes, particularly from error tracebacks.
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    # Also omit standalone Matplotlib Figure representations.
    return "".join(
        line for line in text.splitlines(keepends=True)
        if not re.fullmatch(r"<Figure\b.*>", line.strip())
    )


def image_to_html(output):
    """Embed one image per output, preferring PNG when both formats exist."""
    data = output.get("data", {})
    for mime in IMAGE_MIMETYPES:
        if mime in data:
            payload = data[mime]
            if mime == "image/svg+xml":
                payload = base64.b64encode(payload.encode("utf-8")).decode("ascii")
            else:
                payload = "".join(payload.split())
            return (
                f'<img src="data:{mime};base64,{payload}" alt="Plot output" '
                'style="max-width: min(100%, 1000px); height: auto;" />'
            )
    return ""


def cell_to_markdown(cell):
    if "notebook_setup" in cell.get("metadata", {}).get("tags", []):
        return ""
    source = cell["source"]
    if cell["cell_type"] == "markdown":
        return source
    if cell["cell_type"] != "code" or not source.strip():
        return ""

    body = source.rstrip("\n")
    output = outputs_text(cell.get("outputs", []))
    if output:
        comments = "\n".join(
            "# " + line if line else "#" for line in output.splitlines()
        )
        body += "\n" + comments

    # Avoid closing the Markdown fence if the code contains backticks.
    fence = "```"
    while fence in body:
        fence += "`"
    code_block = f"{fence}python\n{body}\n{fence}"
    images = [image_to_html(output) for output in cell.get("outputs", [])]
    return "\n\n".join([code_block] + [image for image in images if image])


def main():
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(scripts_dir, "..", "docs")
    source_dir = os.path.join(docs_dir, "_srcnbs")
    if not os.path.isdir(source_dir):
        raise SystemExit(f"Notebook directory not found: {source_dir}")

    for path in sorted(glob.glob(os.path.join(source_dir, "*.ipynb"))):
        notebook = nbformat.read(path, as_version=4)
        blocks = [cell_to_markdown(cell) for cell in notebook.cells]
        markdown = "\n\n".join(block for block in blocks if block) + "\n"
        name = os.path.splitext(os.path.basename(path))[0] + ".md"
        destination = os.path.join(docs_dir, name)
        with open(destination, "w", encoding="utf-8", newline="\n") as file:
            file.write(markdown)
        print(f"{os.path.basename(path)} -> docs/{name}")


if __name__ == "__main__":
    main()
