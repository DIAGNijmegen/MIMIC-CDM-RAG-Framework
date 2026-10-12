#!/usr/bin/env python
"""
Batch convert PDFs -> Markdown with Docling.

Usage
-----
python pdf_to_md_ocr.py \
    /path/to/input_pdfs \
    /path/to/output_md \
    --ocr \
    --ocr-engine rapidocr \
    --ocr-backend torch \
    --ocr-lang en \
    --generate-images \
    --image-mode referenced \
    --add-title \
    --artifacts-path /shared/docling_artifacts

Notes
-----
- Requires:
    pip install "docling[all]" transformers huggingface_hub
    # For RapidOCR + ONNX runtime (if you want the ONNX backend):
    pip install rapidocr_onnxruntime
"""

import argparse
import html
import logging
import os
import re
from pathlib import Path
from typing import List, Optional

from huggingface_hub import snapshot_download

from hierarchical.postprocessor import ResultPostprocessor

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    RapidOcrOptions,
    TableFormerMode,
)
from docling_core.types.doc import ImageRefMode
from docling_core.types.doc.document import ContentLayer
from docling_core.transforms.serializer.base import SerializationResult
from docling_core.transforms.serializer.markdown import (
    MarkdownDocSerializer,
    MarkdownParams,
    MarkdownTableSerializer,
)

log = logging.getLogger(__name__)


class CleanMarkdownTableSerializer(MarkdownTableSerializer):
    """
    Custom table serializer that:
    - Uses Docling's standard Markdown table serialization.
    - Then post-processes each table line to:
        * collapse excessive spaces,
        * shrink dot-leader patterns (' . . . . . ' / '.....') to a single '...',
        * shrink long dash runs ('--------') to '---' (Markdown's minimal valid header / rule).
    """

    # precompiled patterns for speed & clarity
    DOT_SPACED_PATTERN = re.compile(r"(?:\.\s){3,}")  # ' . . . . ' (3+ spaced dots)
    DOT_RUN_PATTERN = re.compile(r"\.{4,}")           # '......' (4+ consecutive dots)
    DASH_RUN_PATTERN = re.compile(r"-{4,}")           # '----' or more dashes
    MULTISPACE_PATTERN = re.compile(r"[ \t]{2,}")     # 2+ spaces or tabs

    @classmethod
    def _clean_leaders(cls, text: str) -> str:
        """
        Clean dot-leaders and dash-leaders inside a table row.
        """
        # Collapse spaced dot leaders: " . . . . . " → " ..."
        text = cls.DOT_SPACED_PATTERN.sub(" ... ", text)
        # Collapse long dot runs: "........" → "..."
        text = cls.DOT_RUN_PATTERN.sub("...", text)
        # Collapse long dash runs: "--------" → "---"
        text = cls.DASH_RUN_PATTERN.sub("---", text)
        return text

    @classmethod
    def _clean_whitespace(cls, text: str) -> str:
        """
        Collapse multiple spaces/tabs while keeping one space for readability.
        """
        text = cls.MULTISPACE_PATTERN.sub(" ", text)
        return text.rstrip()

    def serialize(
        self,
        *,
        item,
        doc_serializer,
        doc,
        **kwargs,
    ) -> SerializationResult:
        # Let the base MarkdownTableSerializer create the table (and captions)
        base_res = super().serialize(
            item=item,
            doc_serializer=doc_serializer,
            doc=doc,
            **kwargs,
        )

        if not base_res.text:
            return base_res

        cleaned_lines: list[str] = []
        for line in base_res.text.splitlines():
            if "|" in line:
                # Only modify actual table rows/separator lines
                line = self._clean_leaders(line)
                line = self._clean_whitespace(line)
            cleaned_lines.append(line)

        base_res.text = "\n".join(cleaned_lines)
        return base_res


def build_pdf_converter(
    artifacts_path: Optional[Path],
    enable_ocr: bool,
    ocr_lang: List[str],
    force_full_page_ocr: bool,
    enable_tables: bool,
    generate_images: bool,
    images_scale: float,
    use_rapidocr: bool,
    rapidocr_backend: str = "torch",
) -> DocumentConverter:
    """Configure a Docling DocumentConverter for PDF -> DoclingDocument."""
    pipeline_options = PdfPipelineOptions()

    # Where Docling caches its models (layout, table, OCR, etc.)
    if artifacts_path is not None:
        pipeline_options.artifacts_path = str(artifacts_path)

    # Use GPU for layout / tableformer if available (Docling will fall back if needed).
    pipeline_options.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.CUDA
    )

    # Table structure (recommended ON for RAG)
    pipeline_options.do_table_structure = enable_tables
    if enable_tables and getattr(pipeline_options, "table_structure_options", None):
        ts_opts = pipeline_options.table_structure_options
        # High-quality table model
        ts_opts.mode = TableFormerMode.ACCURATE
        # Let TableFormer define the cell layout directly
        ts_opts.do_cell_matching = False

    # OCR configuration
    if enable_ocr:
        pipeline_options.do_ocr = True

        if use_rapidocr:
            ocr_options: Optional[RapidOcrOptions] = None

            if rapidocr_backend == "onnx":
                # ONNX backend: point to ONNX models from Hugging Face
                if artifacts_path is not None:
                    rapidocr_dir = artifacts_path / "rapidocr_models"
                    download_path = snapshot_download(
                        repo_id="SWHL/RapidOCR",
                        local_dir=str(rapidocr_dir),
                        local_dir_use_symlinks=False,
                    )
                else:
                    download_path = snapshot_download(repo_id="SWHL/RapidOCR")

                det_model_path = os.path.join(
                    download_path, "PP-OCRv4", "en_PP-OCRv3_det_infer.onnx"
                )
                rec_model_path = os.path.join(
                    download_path, "PP-OCRv4", "ch_PP-OCRv4_rec_server_infer.onnx"
                )
                cls_model_path = os.path.join(
                    download_path, "PP-OCRv3", "ch_PP-OCRv4_rec_server_infer.onnx"
                )

                ocr_options = RapidOcrOptions(
                    det_model_path=det_model_path,
                    rec_model_path=rec_model_path,
                    cls_model_path=cls_model_path,
                )

            elif rapidocr_backend == "torch":
                # Torch backend: RapidOCR manages its own .pth models and uses GPU
                ocr_options = RapidOcrOptions(backend="torch")

            else:
                raise ValueError(f"Unknown RapidOCR backend: {rapidocr_backend}")

            # Propagate generic options
            if ocr_options is not None:
                if force_full_page_ocr and hasattr(ocr_options, "force_full_page_ocr"):
                    ocr_options.force_full_page_ocr = True
                if ocr_lang and hasattr(ocr_options, "lang"):
                    ocr_options.lang = ocr_lang

                pipeline_options.ocr_options = ocr_options

        else:
            # Let Docling pick an OCR engine; pass generic options only
            ocr_opts = getattr(pipeline_options, "ocr_options", None)
            if ocr_opts is not None:
                if ocr_lang and hasattr(ocr_opts, "lang"):
                    ocr_opts.lang = ocr_lang
                if force_full_page_ocr and hasattr(ocr_opts, "force_full_page_ocr"):
                    ocr_opts.force_full_page_ocr = True

    # Images / figures
    if generate_images:
        pipeline_options.generate_page_images = True
        pipeline_options.generate_picture_images = True
        pipeline_options.images_scale = images_scale

    format_options = {
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }

    return DocumentConverter(format_options=format_options)


def normalize_markdown_title(
    md_path: Path,
    title: str,
    ensure_top_level: bool = True,
) -> None:
    """
    Post-process a Markdown file to enforce a clean first-level title.

    - Ensures the first non-empty line is `# {title}`.
    - If that first line is already a heading, it normalizes it to one `#`.
    """
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    # Find first non-empty line
    first_idx = None
    for i, line in enumerate(lines):
        if line.strip():
            first_idx = i
            break

    if first_idx is None:
        # Empty document, just write a title
        md_path.write_text(f"# {title}\n", encoding="utf-8")
        return

    first_line = lines[first_idx]

    if first_line.lstrip().startswith("#"):
        # Strip all leading '#' and normalize
        content = first_line.lstrip("#").strip()
        if not content:
            content = title
        if ensure_top_level:
            lines[first_idx] = f"# {content}"
        else:
            # Keep original heading level, but ensure text is title if empty
            lines[first_idx] = first_line if content != "" else f"# {title}"
    else:
        # Insert a new top-level heading before the first content line
        new_lines = []
        new_lines.append(f"# {title}")
        new_lines.append("")  # blank line
        new_lines.extend(lines)
        lines = new_lines

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def infer_doc_title(doc, fallback: str) -> str:
    """
    Try to infer a sensible title from the DoclingDocument, with a file-based fallback.
    """
    # Try explicit document name
    name = getattr(doc, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()

    # Try origin filename (if coming from a file)
    origin = getattr(doc, "origin", None)
    if origin is not None:
        origin_filename = getattr(origin, "filename", None)
        if isinstance(origin_filename, str) and origin_filename.strip():
            return Path(origin_filename).stem

    # Fallback to whatever was passed in (typically the PDF stem)
    return fallback


def clean_markdown_entities(md_text: str) -> str:
    """
    Decode common HTML entities in the Markdown text so that things like
    '&lt;' become '<', '&gt;' become '>', etc.

    Also normalizes non-breaking spaces to regular spaces.
    """
    # Decode standard HTML entities (&lt;, &gt;, &amp;, &le;, &ge;, &nbsp;, ...)
    md_text = html.unescape(md_text)

    # Replace non-breaking spaces with regular spaces so they don't cause odd spacing
    md_text = md_text.replace("\u00a0", " ")

    return md_text


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert PDFs to Markdown using Docling."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Input folder containing PDF files.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output folder where Markdown files will be written.",
    )
    parser.add_argument(
        "--artifacts-path",
        type=Path,
        default=None,
        help="Optional path for Docling artifacts cache (models, etc.).",
    )

    # Table structure
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Disable table structure extraction (not recommended for RAG).",
    )

    # OCR options
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR for scanned PDFs / images.",
    )
    parser.add_argument(
        "--ocr-lang",
        nargs="+",
        default=["en"],
        help="OCR language(s), e.g. --ocr-lang en fr.",
    )
    parser.add_argument(
        "--force-full-page-ocr",
        action="store_true",
        help=(
            "If supported by the OCR engine, OCR the full page image even if "
            "text is embedded."
        ),
    )
    parser.add_argument(
        "--ocr-engine",
        choices=["auto", "rapidocr"],
        default="auto",
        help="OCR engine to use when --ocr is enabled (default: auto).",
    )
    parser.add_argument(
        "--ocr-backend",
        choices=["onnx", "torch"],
        default="torch",
        help="Backend to use for RapidOCR (default: torch, recommended for GPU).",
    )

    # Image export options
    parser.add_argument(
        "--generate-images",
        action="store_true",
        help="Generate page and picture images and reference them from Markdown.",
    )
    parser.add_argument(
        "--images-scale",
        type=float,
        default=2.0,
        help="Scale factor for generated images (default: 2.0).",
    )
    parser.add_argument(
        "--image-mode",
        choices=["placeholder", "embedded", "referenced"],
        default="placeholder",
        help=(
            "How images are represented in Markdown: "
            "placeholder (no actual image), embedded (base64), or referenced (files on disk)."
        ),
    )

    # Title / heading normalization
    parser.add_argument(
        "--add-title",
        action="store_true",
        help="Insert/normalize a '# Title' heading at the top of each Markdown file.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    log.info("Building Docling converter for PDFs...")

    converter = build_pdf_converter(
        artifacts_path=args.artifacts_path,
        enable_ocr=args.ocr,
        ocr_lang=args.ocr_lang,
        force_full_page_ocr=args.force_full_page_ocr,
        enable_tables=not args.no_tables,
        generate_images=args.generate_images,
        images_scale=args.images_scale,
        use_rapidocr=(args.ocr_engine == "rapidocr"),
        rapidocr_backend=args.ocr_backend,
    )

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Optional: where images will be written when using REFERENCED mode
    artifacts_dir = output_dir / "media"
    if args.image_mode == "referenced":
        artifacts_dir.mkdir(parents=True, exist_ok=True)
    else:
        artifacts_dir = None

    image_mode = {
        "placeholder": ImageRefMode.PLACEHOLDER,
        "embedded": ImageRefMode.EMBEDDED,
        "referenced": ImageRefMode.REFERENCED,
    }[args.image_mode]

    pdf_files = sorted(input_dir.rglob("*.pdf"))
    if not pdf_files:
        log.warning("No PDFs found in %s", input_dir)
        return

    log.info("Found %d PDF(s). Converting...", len(pdf_files))

    for pdf_path in pdf_files:
        rel = pdf_path.relative_to(input_dir)
        doc_stem = pdf_path.stem
        out_md_path = output_dir / rel.with_suffix(".md")
        out_md_path.parent.mkdir(parents=True, exist_ok=True)

        log.info("Converting %s -> %s", pdf_path, out_md_path)

        try:
            result = converter.convert(str(pdf_path))

            # Infer and fix heading hierarchy *before* you grab dl_doc
            ResultPostprocessor(result, source=str(pdf_path)).process()

            dl_doc = result.document

            serializer = MarkdownDocSerializer(
                doc=dl_doc,
                table_serializer=CleanMarkdownTableSerializer(),
                params=MarkdownParams(
                    layers={ContentLayer.BODY},
                    image_mode=image_mode,
                    image_placeholder="<!-- image -->",
                ),
            )

            ser_result = serializer.serialize()
            md_text = ser_result.text
            md_text = clean_markdown_entities(md_text)
            out_md_path.write_text(md_text, encoding="utf-8")

            # Optional title normalization
            if args.add_title:
                title = infer_doc_title(dl_doc, fallback=doc_stem)
                normalize_markdown_title(out_md_path, title=title)

        except Exception as exc:  # noqa: BLE001
            log.exception("Failed to convert %s: %s", pdf_path, exc)

    log.info("Done.")


if __name__ == "__main__":
    main()
