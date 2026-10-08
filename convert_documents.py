#!/usr/bin/env python3
"""
Batch PDF to Markdown Converter with Image and Table Extraction
Using PyMuPDF4LLM for state-of-the-art layout, math, and figure preservation.
"""

import os
import glob
import re
import pathlib
import pymupdf
import pymupdf4llm
from pymupdf4llm.helpers import utils

# Monkey-patch utils.md_path to ensure target directories always exist,
# preventing any filesystem errors regardless of special characters in filenames.
_orig_md_path = utils.md_path
def _safe_md_path(folder: str, filename: str):
    md_ref, save_path = _orig_md_path(folder, filename)
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    return md_ref, save_path
utils.md_path = _safe_md_path

def sanitize_name(name: str) -> str:
    """Sanitize string to avoid filename and markdown link issues."""
    return (
        name.replace("(", "-")
        .replace(")", "-")
        .replace("[", "-")
        .replace("]", "-")
        .replace(" ", "_")
    )

def parse_paper_title(pdf_path):
    """Extract paper title from PDF metadata or the first page."""
    try:
        doc = pymupdf.open(pdf_path)
        meta_title = doc.metadata.get("title", "").strip()
        if meta_title and len(meta_title) > 5 and not meta_title.lower().endswith(".pdf"):
            return meta_title
        
        # Fallback to reading first lines from page 0
        first_page_text = doc[0].get_text("text").split("\n")
        lines = [line.strip() for line in first_page_text if line.strip()]
        title_lines = []
        for line in lines:
            if any(skip in line.lower() for skip in ["published as", "arxiv:", "proceedings", "conference", "under review"]):
                continue
            title_lines.append(line)
            if len(title_lines) >= 2:
                break
        if title_lines:
            return " ".join(title_lines)
    except Exception as e:
        print(f"  Warning extracting title: {e}")
    return pathlib.Path(pdf_path).stem

def convert_pdf_to_md(pdf_path, papers_dir):
    """Convert a single PDF to Markdown and extract images into a same-level folder."""
    pdf_path = pathlib.Path(pdf_path).resolve()
    papers_dir = pathlib.Path(papers_dir).resolve()
    stem = pdf_path.stem
    
    # Image directory on the same level: Documents/Papers/{sanitized_stem}_images
    img_dir_name = f"{sanitize_name(stem)}_images"
    img_dir_full = papers_dir / img_dir_name
    img_dir_full.mkdir(parents=True, exist_ok=True)
    
    md_output_path = papers_dir / f"{stem}.md"
    
    print(f"\n==================================================")
    print(f"Processing: {pdf_path.name}")
    doc = pymupdf.open(str(pdf_path))
    num_pages = len(doc)
    print(f"  Total pages: {num_pages}")
    
    # Change working directory temporarily to papers_dir so image links in markdown
    # are generated cleanly as relative paths: {img_dir_name}/image.png
    original_cwd = os.getcwd()
    try:
        os.chdir(str(papers_dir))
        md_text = pymupdf4llm.to_markdown(
            doc=str(pdf_path),
            write_images=True,
            image_path=img_dir_name,
            image_format="png",
            dpi=150,
            show_progress=True
        )
    finally:
        os.chdir(original_cwd)
    
    # Normalize any Windows backslashes in image markdown tags to standard forward slashes
    md_text = re.sub(r'!\[(.*?)\]\((.*?)\)', lambda m: f'![{m.group(1)}]({m.group(2).replace(chr(92), "/")})', md_text)
    
    # Write output markdown with UTF-8 encoding
    md_output_path.write_text(md_text, encoding="utf-8")
    
    # Count generated images
    extracted_images = list(img_dir_full.glob("*.png"))
    print(f"  Generated: {md_output_path.name} ({len(md_text):,} characters)")
    print(f"  Extracted images: {len(extracted_images)} in {img_dir_name}/")
    
    title = parse_paper_title(pdf_path)
    return {
        "stem": stem,
        "pdf_name": pdf_path.name,
        "md_name": md_output_path.name,
        "img_dir": img_dir_name,
        "num_pages": num_pages,
        "chars": len(md_text),
        "num_images": len(extracted_images),
        "title": title
    }

def generate_index_readme(papers_dir, results):
    """Generate a README.md index in the papers directory."""
    papers_dir = pathlib.Path(papers_dir)
    readme_path = papers_dir / "README.md"
    
    lines = [
        "# Ethics in AI - Research Papers Collection",
        "",
        "This directory contains parsed Markdown versions of academic research papers on AI alignment, sycophancy, and evaluation, complete with extracted figures, diagrams, and formatted tables.",
        "",
        "## Documents Index",
        "",
        "| Document | Title | Pages | Extracted Images | Markdown File |",
        "| :--- | :--- | :---: | :---: | :--- |"
    ]
    
    for r in results:
        doc_link = f"[{r['pdf_name']}](./{r['pdf_name']})"
        md_link = f"[{r['md_name']}](./{r['md_name']})"
        img_info = f"{r['num_images']} ([folder](./{r['img_dir']}/))" if r['num_images'] > 0 else "0"
        lines.append(f"| {doc_link} | **{r['title']}** | {r['num_pages']} | {img_info} | {md_link} |")
    
    lines.extend([
        "",
        "## Directory Structure",
        "",
        "```text",
        "Documents/Papers/",
    ])
    for r in results:
        lines.append(f"├── {r['pdf_name']}")
        lines.append(f"├── {r['md_name']}")
        lines.append(f"├── {r['img_dir']}/")
    lines.extend([
        "└── README.md",
        "```",
        "",
        "## Parsing Details",
        "- **Parser Engine**: PyMuPDF4LLM with PyMuPDF Layout (ONNX layout analyzer)",
        "- **Image Resolution**: 150 DPI PNG",
        "- **Image Linking**: Relative Markdown syntax `![...](<paper_stem>_images/...)` compatible with GitHub, Obsidian, and VS Code.",
        "- **Encoding**: UTF-8",
        ""
    ])
    
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nCreated master index: {readme_path}")

def main():
    papers_dir = pathlib.Path(__file__).parent / "Documents" / "Papers"
    if not papers_dir.exists():
        print(f"Directory not found: {papers_dir}")
        return
    
    pdf_files = sorted(papers_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {papers_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF file(s) in {papers_dir.relative_to(pathlib.Path.cwd()) if papers_dir.is_relative_to(pathlib.Path.cwd()) else papers_dir}")
    
    results = []
    for pdf_file in pdf_files:
        res = convert_pdf_to_md(pdf_file, papers_dir)
        results.append(res)
    
    generate_index_readme(papers_dir, results)
    print("\n==================================================")
    print("All documents successfully parsed and organized!")

if __name__ == "__main__":
    main()
