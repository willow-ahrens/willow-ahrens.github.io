import bibtexparser
from bibtexparser.middlewares import SeparateCoAuthors, SplitNameParts, MergeNameParts, MonthIntMiddleware
import os
import datetime
import calendar

def format_authors(authors):
    """Format authors as 'first middle last, first middle last, and first middle last'."""
    if len(authors) > 1:
        return ', '.join(authors[:-1]) + ', and ' + authors[-1]
    return authors[0]

def published_date(entry):
    entry = {field.key: field.value for field in entry.fields}
    return datetime.date(int(entry['year']), entry.get('month', 1), entry.get('day', 1))

def format_markdown(entry):
    """Format a single BibTeX entry as Markdown."""
    title = entry['title'].replace('{', '').replace('}', '')
    authors = format_authors(entry['author'])
    year = entry['year']
    if 'month' in entry:
        month = calendar.month_name[entry['month']] + ' '
    else:
        month = ''
    if 'doi' in entry:
        url = f"https://doi.org/{entry['doi']}"
    else:
        url = entry["url"]
    if 'journal' in entry:
        venue = entry['journal']
    elif 'series' in entry:
        venue = entry['series']
    elif 'publisher' in entry:
        venue = entry['publisher']
    elif entry.entry_type == 'mastersthesis':
        venue = "Master's Thesis, " + entry['school']
    elif entry.entry_type == 'techreport':
        venue = entry['institution']
    elif 'publication' in entry:
        venue = entry['publication']
    else:
        venue = 'Unknown Venue'
        print("Unknown venue for entry", entry['ID'])
    venue = venue.replace('{', '').replace('}', '')
        
    #arxiv = entry.get('note', '').split(' ')[0] if 'arXiv' in entry.get('note', '') else None
    filename = os.path.join('assets', 'documents', entry['ID'] + '.pdf')

    markdown = f"### {title}\n"
    markdown += f"{authors}, “{title},” {venue}, {month}{year}.\n"
    markdown += f"[Link]({url})."
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.isfile(os.path.join(root_dir, filename)):
        markdown += f" [Download](/{filename})."
    else:
        print(f"File {filename} not found")
    markdown += "\n"
    return markdown


def convert_bibtex_to_markdown(bibtex_file, output_file):
    """Convert a BibTeX file to Markdown."""
    # Define middleware chain for parsing
    layers = [
        SeparateCoAuthors(),
        SplitNameParts(),
        MergeNameParts("first"),
        MonthIntMiddleware(),
    ]

    bib_database = bibtexparser.parse_file(bibtex_file, append_middleware=layers)

    #with open(bibtex_file, 'r') as bibfile:
    #    # Load the BibTeX database with middleware chain
    #    bib_database = bibtexparser.parse_string(bibfile, middlewares=middlewares)

    sorted_entries = sorted(bib_database.entries, key=published_date, reverse=True)

    markdown_entries = list(map(format_markdown, sorted_entries))

    with open(output_file, 'w') as outfile:
        outfile.write(
"""---
permalink: /publications/
title: "Publications"
---

[Curriculum Vitae](/assets/documents/willow_ahrens_cv.pdf)

""")
        outfile.write("\n".join(markdown_entries))
    print(f"Markdown bibliography written to {output_file}")


# Example usage
convert_bibtex_to_markdown('Website.bib', 'publications.md')