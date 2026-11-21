"""Label the keyword-precision sheet one post at a time, in a terminal.

Run it from the project folder (where the database and the sheet are):

    python label_posts.py

Each post is shown whole: the title, a link to the post on Reddit, the link it points
to (for link posts), and the body. Type y, n or ? and press Enter, with a note number
after a space ("n 1") or a note in your own words ("n about the flu shot only").
m shows the rest of a long post; q stops. Every answer is saved to the CSV at once,
so you can stop at any point and run the script again to continue where you left off.

This does the same job as the old label_sheet() cell in the notebook. It runs in a
terminal because Jupyter's input box kept disappearing between posts.
"""
import os
import re
import sqlite3
import sys
import textwrap
from pathlib import Path

import pandas as pd

#db and sheet sit in same folder as this script
FOLDER = Path(__file__).resolve().parent
DB_PATH = FOLDER / "vaccine_discourse.db"
SHEET_PATH = FOLDER / "keyword_precision_labels.csv"

#notes written out in full, so sheet still makes sense months from now
#(table in notebook's "Precision check, part 2" lists them)
#type label and number ("n 1") to save that note, or own sentence after label instead
#{Topic} and {topic} become COVID or vaccines, depending on post's question
NOTES = {
    "y": {"": "{Topic} {is} the main subject of the post.",
          "1": "{Topic} {informs} the context of the post to a great extent, though {it_is} not the main subject.",
          "2": "Judged from the title alone (a link post, or no body text)."},
    "n": {"": "The post is not about {topic}.",
          "1": "{Topic} {is} mentioned only to mark a point in time (\"since covid\", \"pre-pandemic\").",
          "2": "{Topic} {is} mentioned in passing and {has} nothing to do with the point of the post.",
          "3": "The matched word means something else here (e.g. a \"shot\" in basketball)."},
    "?": {"1": "Too little text to decide.",
          "2": "Unsure; set aside to discuss."},
}

def fill_note(template, question):
    """Put COVID or vaccines into a note, with verbs that agree."""
    if "COVID" in question:
        words = {"Topic": "COVID", "topic": "COVID", "is": "is", "informs": "informs",
                 "it_is": "it is", "has": "has"}
    else:
        words = {"Topic": "Vaccines", "topic": "vaccines", "is": "are", "informs": "inform",
                 "it_is": "they are", "has": "have"}
    return template.format(**words)

def print_note_key(question):
    print('Notes (type the label and a number, e.g. "n 1"; or write your own after the label):')
    for label, notes in NOTES.items():
        for number, template in notes.items():
            key = f"{label} {number}".strip()
            print(f"   {key:4s} = {fill_note(template, question)}")

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")

def ask_for_label(question, body, max_chars, width):
    """Ask until the answer is valid. Returns (label, note), or None if the user typed q."""
    while True:
        answer = input("> ").strip()
        if answer.lower() == "q":
            return None
        if answer.lower() == "m":
            if len(body) > max_chars:
                print("\n" + textwrap.fill(body[max_chars:], width) + "\n")
            else:
                print("That is the whole post.")
            continue
        label, _, note = answer.partition(" ")
        label = label.lower()
        if label not in ("y", "n", "?"):
            print("Please type y, n or ? (or m for the rest of a long post, q to stop).")
            continue
        first_word, _, rest = note.strip().partition(" ")
        if first_word.isdigit():
            if first_word not in NOTES[label]:
                print(f'There is no note "{label} {first_word}". Please answer again.')
                continue
            note = fill_note(NOTES[label][first_word], question)
            if rest.strip():
                note += " Also: " + rest.strip() #extra words typed after the number
        elif note.strip() == "" and "" in NOTES[label]:
            note = fill_note(NOTES[label][""], question)
        return label, note.strip()

def label_sheet(width=100, max_chars=2500):
    sheet = pd.read_csv(SHEET_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    conn = sqlite3.connect(DB_PATH)
    to_label = sheet.index[sheet["label"].str.strip() == ""].tolist()
    n_done_before = len(sheet) - len(to_label)

    for count, i in enumerate(to_label, start=1):
        row = sheet.loc[i]
        title, body, url, is_self = conn.execute(
            "SELECT title, selftext, url, is_self FROM reddit_posts WHERE reddit_id = ?",
            (row["reddit_id"],),
        ).fetchone()
        body = " ".join((body or "").split()) #collapse line breaks and extra spaces
        matched = re.search(r">>(.*?)<<", row["context"])

        clear_screen() #show 1 post at a time
        print(f"Post {n_done_before + count} of {len(sheet)}    rule: {row['keyword_group']} / {row['keyword']}    r/{row['subreddit']}")
        print(f"QUESTION: Is this post {row['question'].upper()}")
        if matched:
            print(f'Matched: "{matched.group(1)}"')
        print("\nTITLE:", textwrap.fill(title or "", width))
        print("POST: ", row["permalink"], "  (the post on Reddit, with its comments)")
        if not is_self:
            print("LINK: ", url, "  (a link post: the linked page itself was not collected)")
        if body:
            print("\n" + textwrap.fill(body[:max_chars], width))
            if len(body) > max_chars:
                print(f"\n... ({len(body) - max_chars:,} more characters: type m to see the rest)")
        else:
            print("\n(no body text)")
        print(f"\nIs this post {row['question']}   y / n / ?   (note after a space; m = rest of a long post; q = stop)")
        print_note_key(row["question"])

        answer = ask_for_label(row["question"], body, max_chars, width)
        if answer is None:
            print("Stopped. Run the script again to continue.")
            return
        sheet.at[i, "label"], sheet.at[i, "note"] = answer
        sheet.to_csv(SHEET_PATH, index=False, encoding="utf-8-sig") #save after every answer

    clear_screen()
    print("All", len(sheet), "posts labeled. Run the precision cells in the notebook next.")

if __name__ == "__main__":
    #some Windows terminals can't print every char (emoji, some scripts)
    #show placeholder for those instead of crashing
    sys.stdout.reconfigure(errors="replace")
    try:
        label_sheet()
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Every answer before this one was saved; run the script again to continue.")
