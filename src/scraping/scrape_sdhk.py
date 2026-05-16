#!/usr/bin/env python3
"""
Scrape SDHK charters with a CAPTCHA cookie.

Usage:
    .venv/bin/python3 src/scraping/scrape_sdhk.py \\
        --cookie "altcha_verified=..." \\
        --ids-file path/to/charter_ids.json \\
        --output data/raw/scraped.json

The --ids-file must be a JSON list of charter IDs (strings or ints).
Resumes automatically if interrupted — skips already-scraped IDs.
"""

import json
import re
import ssl
import sys
import time
import urllib.request
import argparse


# ---------------------------------------------------------------------------
# HTML parsing (adapted from scrape_pilot_letters.py)
# ---------------------------------------------------------------------------

def extract_between(text, start_marker, end_marker):
    """Extract text between two markers."""
    p0 = text.find(start_marker)
    if p0 == -1:
        return ''
    p1 = text.find(end_marker, p0 + len(start_marker))
    if p1 == -1:
        return ''
    return text[p0 + len(start_marker):p1].strip()


def clean_html(text):
    """Remove HTML tags and clean up text (for summaries)."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    text = text.replace('&nbsp;', ' ')
    # Remove quotation marks but keep content
    for ch in ['"', '\u201c', '\u201d', '\u201e', '\u00ab', '\u00bb',
               "'", '\u2018', '\u2019']:
        text = text.replace(ch, '')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def clean_transcription(trtxt):
    """Clean transcription/edition text — remove footnotes, HTML, fix spacing."""
    # Remove hidden footnote spans
    trtxt = re.sub(r'<span class="hidden"[^>]*>.*?</span></span>', '', trtxt, flags=re.DOTALL)
    # Remove footnote markers (sup tags with note class)
    trtxt = re.sub(r'<sup[^>]*class="note"[^>]*>[^<]*</sup>', '', trtxt)
    trtxt = re.sub(r'<sup[^>]*data-target="#not-[^"]*"[^>]*>[^<]*</sup>', '', trtxt)
    # Newlines → spaces
    trtxt = re.sub(r'[\r\n]', ' ', trtxt)
    # Paragraph tags
    trtxt = re.sub(r'</p>', '', trtxt)
    trtxt = re.sub(r'<p>', '\n\n', trtxt)
    trtxt = re.sub(r'<p\s*/>', '', trtxt)
    # Remove <em> tags, keep content
    trtxt = re.sub(r'<em>([^<]*)</em>', r'\1', trtxt)
    # Remove remaining sup/span/a tags, keep content
    trtxt = re.sub(r'<sup[^>]*>([^<]*)</sup>', r'\1', trtxt)
    trtxt = re.sub(r'<sup>', '', trtxt)
    trtxt = re.sub(r'</sup>', '', trtxt)
    trtxt = re.sub(r'<span[^>]*>', '', trtxt)
    trtxt = re.sub(r'</span>', '', trtxt)
    trtxt = re.sub(r'<a[^>]*>', '', trtxt)
    trtxt = re.sub(r'</a>', '', trtxt)
    # Remove any remaining HTML tags
    trtxt = re.sub(r'<[^>]+>', '', trtxt)
    # Clean up special markers
    trtxt = re.sub(r' [*][)] ', ' ', trtxt)
    # Remove parentheses WITH contents (footnote references)
    trtxt = re.sub(r'\([^)]*\)', '', trtxt)
    # Remove bracket CHARACTERS, keep contents (reconstructed letters)
    trtxt = trtxt.replace('[', '')
    trtxt = trtxt.replace(']', '')
    # Decode HTML entities
    trtxt = trtxt.replace('&amp;', '&')
    trtxt = trtxt.replace('&lt;', '<')
    trtxt = trtxt.replace('&gt;', '>')
    # Remove invisible/special characters
    for ch in ['\u00ad', '\u200b', '\u200c', '\u200d', '\ufeff']:
        trtxt = trtxt.replace(ch, '')
    # Normalize special spaces → regular space
    for ch in ['\u2006', '\u2007', '\u2008', '\u2009', '\u200a', '\u00a0']:
        trtxt = trtxt.replace(ch, ' ')
    # Collapse multiple spaces
    trtxt = re.sub(r' +', ' ', trtxt)
    return trtxt.strip()


def extract_summary(html_content):
    """Extract Innehåll (modern Swedish summary/regest)."""
    summary = extract_between(html_content, '<h5>Innehåll</h5><div class="sdhk-innehall">', '</div>')
    if summary:
        summary = re.sub(r'<span class="sdhk[-]utfardare">(.*?)</span>[,]?', r'\1', summary)
        summary = clean_html(summary)
    return summary


def extract_edition(html_content):
    """Extract Brevtext (Old Swedish transcription/edition)."""
    edition = extract_between(html_content, '<div class="sdhk-brevtext">', '</div>')
    if edition:
        edition = clean_transcription(edition)
    return edition


def extract_date(html_content):
    """Extract Datering."""
    date_text = extract_between(html_content, '<h5>Datering</h5><span class="sdhk-brevhuvud"><span', '</span>')
    if date_text:
        date_text = re.sub(r'<a .*</a>', '', date_text)
        date_text = re.sub(r'.*?>', '', date_text, 1)
        date_text = clean_html(date_text)
    return date_text


def extract_language(html_content):
    """Extract Språk."""
    lang = extract_between(html_content, '<h5>Språk</h5>', '<')
    return lang.strip() if lang else ''


def extract_place(html_content):
    """Extract Utfärdandeort."""
    place = extract_between(html_content, '<h5>Utfärdandeort</h5>', '<')
    if place:
        place = clean_html(place)
    return place or ''


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def scrape_charter(charter_id, cookie):
    """Scrape a single charter from SDHK using the CAPTCHA cookie."""
    url = f'https://sok.riksarkivet.se/sdhk?SDHK={charter_id}&page=1&postid=sdhk_{charter_id}&tab=post#tab'

    ssl._create_default_https_context = ssl._create_unverified_context

    req = urllib.request.Request(url)
    req.add_header('Cookie', cookie)
    req.add_header('User-Agent', 'Mozilla/5.0 (academic research)')

    try:
        resp = urllib.request.urlopen(req, timeout=30)
        html = resp.read().decode('utf-8')

        # Check if we got CAPTCHA instead of content
        if '<title>Captcha' in html and '<h5>Brevtext</h5>' not in html:
            return {'error': 'CAPTCHA_EXPIRED'}

        sdhk_content = extract_between(html, '<div class="sdhk post">', '<div class="kommentarer">')
        if not sdhk_content:
            return {'error': 'NO_CONTENT'}

        return {
            'Id': charter_id,
            'Date': extract_date(sdhk_content),
            'Place': extract_place(sdhk_content),
            'Language': extract_language(sdhk_content),
            'Summary': extract_summary(sdhk_content),
            'Edition': extract_edition(sdhk_content),
            'source': 'web_scrape_2026'
        }

    except urllib.error.URLError as e:
        return {'error': f'URL_ERROR: {e}'}
    except Exception as e:
        return {'error': f'ERROR: {e}'}


def main():
    parser = argparse.ArgumentParser(description='Scrape SDHK charters')
    parser.add_argument('--cookie', required=True, help='altcha_verified cookie value')
    parser.add_argument('--ids-file', required=True,
                        help='JSON file containing a list of charter IDs to scrape')
    parser.add_argument('--output', default='sdhk_scraped.json', help='Output JSON file')
    parser.add_argument('--delay', type=float, default=1.5, help='Delay between requests (seconds)')
    args = parser.parse_args()

    with open(args.ids_file, 'r', encoding='utf-8') as f:
        charter_ids = [str(cid) for cid in json.load(f)]
    print(f"Loaded {len(charter_ids)} charter IDs from {args.ids_file}")

    print(f"Total charters to scrape: {len(charter_ids)}")

    # Resume support: load already-scraped charters
    results = []
    already_scraped = set()
    try:
        with open(args.output, 'r', encoding='utf-8') as f:
            results = json.load(f)
            already_scraped = {str(r['Id']) for r in results if 'error' not in r}
            print(f"Resuming: {len(already_scraped)} already scraped")
    except FileNotFoundError:
        pass

    remaining = [cid for cid in charter_ids if cid not in already_scraped]
    print(f"Remaining to scrape: {len(remaining)}")

    if not remaining:
        print("All charters already scraped!")
        return

    # Scrape
    success = 0
    errors = 0
    for i, cid in enumerate(remaining):
        print(f"[{len(already_scraped) + i + 1}/{len(charter_ids)}] SDHK {cid}...", end=' ', flush=True)

        result = scrape_charter(cid, args.cookie)

        if 'error' in result:
            if result['error'] == 'CAPTCHA_EXPIRED':
                print("CAPTCHA EXPIRED — cookie no longer valid. Saving progress.")
                break
            print(f"ERROR: {result['error']}")
            errors += 1
        else:
            s_len = len(result.get('Summary', ''))
            e_len = len(result.get('Edition', ''))
            print(f"OK (Summary: {s_len}, Edition: {e_len})")
            results.append(result)
            success += 1

        # Save after every 10 charters (incremental)
        if (i + 1) % 10 == 0:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

        time.sleep(args.delay)

    # Final save
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nDone: {success} scraped, {errors} errors, {len(results)} total saved to {args.output}")


if __name__ == '__main__':
    main()
