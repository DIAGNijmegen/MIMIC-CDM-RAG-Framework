# Convert PDFs to Markdown

Converts PDF files to Markdown files, that can then be manually cleaned.
`docling/pdf_to_md_ocr.py` recursively converts PDFs with Docling, preserving
subfolder paths and cleaning headings and tables. It uses CLI flags, not Hydra YAML.

```bash
python docling/pdf_to_md_ocr.py /path/to/pdfs /path/to/markdown \
  --ocr --ocr-engine rapidocr --ocr-backend torch --ocr-lang en --add-title
```

| Argument | Purpose |
| --- | --- |
| `input_dir`, `output_dir` | Positional input PDF folder and Markdown destination. |
| `--ocr` | Enable OCR. |
| `--ocr-engine` | `auto` (default) or `rapidocr`. |
| `--ocr-backend` | RapidOCR backend: `torch` (default) or `onnx`. |
| `--ocr-lang` | Language list, e.g. `en fr`; defaults to `en`. |
| `--force-full-page-ocr` | OCR the full page when supported by the engine. |
| `--no-tables` | Disable table structure extraction, which is enabled by default. |
| `--generate-images`, `--images-scale` | Generate page/picture images; scale defaults to `2.0`. |
| `--image-mode` | `placeholder` (default), `embedded`, or `referenced`. |
| `--add-title` | Insert or normalize the document's top-level heading. |
| `--artifacts-path` | Optional model/artifact cache directory. |

Use an environment with Docling and the selected OCR backend installed.
Each PDF produces a corresponding `.md` file. Use the resulting paths in
`rag_documents` for [corpus building](build_rag_bd.md) or [inference](run_rag.md).
