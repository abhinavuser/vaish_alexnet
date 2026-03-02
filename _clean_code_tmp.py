import io
import tokenize
from pathlib import Path

files = [
    Path('analyze_dataset.py'),
    Path('augment_data.py'),
    Path('train_alexnet.py'),
    Path('train_final.py'),
    Path('train_with_checkpoints.py'),
    Path('visualize_results.py'),
]

def remove_comments_and_docstrings(src: str) -> str:
    io_obj = io.StringIO(src)
    out = []
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type, token_string, start, end, _ = tok
        start_line, start_col = start
        end_line, end_col = end
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out.append(' ' * (start_col - last_col))

        if token_type == tokenize.COMMENT:
            pass
        elif token_type == tokenize.STRING and prev_toktype in (tokenize.INDENT, tokenize.NEWLINE, tokenize.DEDENT):
            pass
        else:
            out.append(token_string)

        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line

    text = ''.join(out)
    lines = []
    for i, line in enumerate(text.splitlines()):
        if i == 0 and line.startswith('#!'):
            lines.append(line)
            continue
        lines.append(line.rstrip())
    text = '\n'.join(lines) + '\n'

    replacements = {
        'TOMATO LEAVES DATASET ANALYSIS REPORT': 'Tomato leaves dataset analysis report',
        'TRAINING PIPELINE: TOMATO LEAVES DISEASE DETECTION WITH ALEXNET': 'Training pipeline: tomato leaves disease detection with AlexNet',
        'PHASE 1 RESULTS: CROSS-VALIDATION SUMMARY': 'Phase 1 results: cross-validation summary',
        'DETAILED CLASSIFICATION REPORT': 'Detailed classification report',
        'ALEXNET TRAINING ON AUGMENTED TOMATO LEAVES DATASET': 'AlexNet training on augmented tomato leaves dataset',
        'CROSS-VALIDATION SUMMARY': 'Cross-validation summary',
        'DATA AUGMENTATION PIPELINE': 'Data augmentation pipeline',
        'AUGMENTATION COMPLETE!': 'Augmentation complete!',
    }
    for k, v in replacements.items():
        text = text.replace(k, v)

    return text

for file in files:
    if not file.exists():
        continue
    src = file.read_text(encoding='utf-8')
    cleaned = remove_comments_and_docstrings(src)
    file.write_text(cleaned, encoding='utf-8')
    print(f'cleaned: {file}')
